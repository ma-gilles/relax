"""Device-resident PPCA statistics for streamed full fine rotation rows.

Every image of a tile scores the same full fine rotation grid; only its
translation support differs, and that support is the coarse pass-1 significant
set expanded to fine children. This module uploads the tile's coarse support
once and expands the fine pose log-prior on the device inside each rotation
block program, so no per-block host mask, prior or synchronization remains.

Per image tile the work is three sequences of one jitted program per rotation
block, all enqueued without host round trips:

1. score, latent moments and the running top pose and score maximum
   (:func:`_score_block`);
2. the centered pose partition (:func:`_add_block_partition`);
3. expected residuals, augmented ``[mu, W]`` backprojection and posterior
   diagnostics (:func:`_backproject_block`).

The arithmetic is that of :func:`relax.ppca_refinement.dense_dataset.
accumulate_dense_ppca_statistics` with ``collect_residuals=True`` and sparse
pass 2 disabled, and of :func:`relax.ppca_refinement.dense_dataset.
compute_dense_ppca_embeddings`: the same score functions, float32 centered
normalization (``docs/math/vdam_ppca_algorithm.md``), block accumulation order
and top-1 tie rule. Tests compare both against the host-mask reference.
The formulation is section 14 of ``docs/math/vdam_ppca_algorithm.md``.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from recovar.core.configs import ForwardModelConfig
from recovar.ppca.pose_accumulators import AugmentedPPCAStats
from recovar.ppca.triangular import tri_size
from recovar.reconstruction import noise as noise_utils

from relax.helpers.adjoint import batch_adjoint_slice_volume_maybe_windowed
from relax.helpers.half_spectrum import make_half_image_weights, make_shell_indices_half
from relax.ppca_refinement.config import GeometryConfig, ScheduleConfig, ScoringConfig
from relax.ppca_refinement.dense_dataset import (
    DensePPCAEmbeddings,
    _latent_covariance_trace_from_packed_moments,
    _project_augmented_half_volumes,
    prepare_dense_ppca_dataset_inputs,
    prepare_dense_ppca_image_batch,
)
from relax.ppca_refinement.engine import (
    _enforce_augmented_x0,
    backproject_moment_images,
    dense_pose_ppca_score_with_moments_blocked,
    dense_pose_ppca_score_with_moments_factor_once,
    pose_moment_images,
)
from relax.ppca_refinement.pose_selection import top_p_from_score_block
from relax.ppca_refinement.residual_statistics import full_float32, residual_statistics_from_moment_images

FULL_ROW_ENGINE = "full_row_device_resident"

_SCORE_FUNCTIONS = {
    "blocked": dense_pose_ppca_score_with_moments_blocked,
    "factor_once": dense_pose_ppca_score_with_moments_factor_once,
}


class _StreamStatic(NamedTuple):
    """Hashable shape and kernel choices shared by every block program."""

    image_shape: tuple[int, int]
    volume_shape: tuple[int, int, int]
    disc_type: str
    projection_max_r: float | None
    backprojection_max_r: float | None
    relion_texture_interp: bool
    use_recon_window: bool
    basis_size: int


class _StreamArrays(NamedTuple):
    """Device operands fixed for one expectation (model, grids, priors)."""

    augmented: jax.Array  # (P, half_volume) complex64 [mu, W_1..W_q]
    rotations: jax.Array  # (R, 3, 3) float32 fine rotation grid
    rotation_log_prior: jax.Array  # (R,) float32
    translation_log_prior: jax.Array  # (T,) float32
    rotation_parent: jax.Array  # (R,) int32 coarse rotation of each fine row
    translation_parent: jax.Array  # (T,) int32 coarse translation of each fine shift
    score_indices: jax.Array | None  # score window in the packed half image
    recon_indices: jax.Array | None  # reconstruction window
    coefficient_noise: jax.Array  # (n_half,) noise variance per half-image pixel
    shift_squared: jax.Array  # (T,) squared fine shift length in px^2


class _TileArrays(NamedTuple):
    """Device operands for one image tile, uploaded once per tile."""

    coarse_mask: jax.Array  # (B, R_coarse, T_coarse) bool significant coarse poses
    Y1: jax.Array
    ctf2: jax.Array
    Y1_recon: jax.Array
    ctf2_recon: jax.Array
    y_norm: jax.Array


class _TopPose(NamedTuple):
    """Running per-image score maximum and first-occurring top-1 pose."""

    center: jax.Array
    score: jax.Array
    rotation: jax.Array
    translation: jax.Array


class _MomentCarry(NamedTuple):
    """Pass-2 accumulators, updated in block order as the host-mask reference."""

    rhs: jax.Array
    lhs_tri: jax.Array
    residual: jax.Array
    residual_power: jax.Array
    embedding: jax.Array
    latent_covariance_trace_sum: jax.Array
    pose_entropy_sum: jax.Array
    offset_second_sum: jax.Array
    n_significant: jax.Array


class FullRowStream(NamedTuple):
    """Prepared inputs of one streamed full-row expectation over image tiles."""

    dataset: object
    arrays: _StreamArrays
    static: _StreamStatic
    forward_config: ForwardModelConfig
    resolved: object
    noise_variance_half: jax.Array
    translations: np.ndarray
    n_coarse_rotations: int
    n_coarse_translations: int
    image_batch_size: int
    blocks: tuple[tuple[jax.Array, int], ...]  # (device start, static size) per rotation block
    rotation_block_size: int


def coarse_support_mask(significant_rows, n_coarse_rotations: int, n_coarse_translations: int) -> np.ndarray:
    """Return ``(B, R_coarse, T_coarse)`` coarse support from packed significant ids.

    ``None`` keeps every coarse pose, as in
    :func:`relax.local.local_layout.build_pass2_hypothesis_layout`.
    """
    coarse = np.zeros((len(significant_rows), int(n_coarse_rotations) * int(n_coarse_translations)), dtype=bool)
    for image, significant in enumerate(significant_rows):
        if significant is None:
            coarse[image] = True
        else:
            coarse[image, np.asarray(significant, dtype=np.int64)] = True
    return coarse.reshape(len(significant_rows), int(n_coarse_rotations), int(n_coarse_translations))


def full_row_pose_log_prior(
    coarse_mask, rotation_parent, translation_parent, rotation_log_prior, translation_log_prior
) -> jax.Array:
    """Expand coarse support to the ``(B, R, T)`` fine pose log-prior on the device.

    A fine pose ``(r, t)`` is supported exactly when its coarse parent
    ``(rotation_parent[r], translation_parent[t])`` was significant. Supported
    poses carry ``rotation_log_prior[r] + translation_log_prior[t]`` in float32
    and unsupported ones ``-inf``, matching ``_per_image_pose_prior_block``.
    """
    mask = jnp.take(jnp.take(coarse_mask, rotation_parent, axis=1), translation_parent, axis=2)
    prior = rotation_log_prior[:, None] + translation_log_prior[None, :]
    return jnp.where(mask, prior[None], -jnp.inf).astype(jnp.float32)


def _block_rows(array, start, size: int):
    return jax.lax.dynamic_slice_in_dim(array, start, size, axis=0)


def _score_window_projection(arrays: _StreamArrays, rotations_block, static: _StreamStatic):
    proj = _project_augmented_half_volumes(
        arrays.augmented,
        rotations_block,
        static.image_shape,
        static.volume_shape,
        static.disc_type,
        max_r=static.projection_max_r,
        relion_texture_interp=static.relion_texture_interp,
    )
    return proj if arrays.score_indices is None else proj[:, :, arrays.score_indices]


@partial(jax.jit, static_argnames=("static", "block_size", "score_kind", "keep_second_moment"))
def _score_block(arrays, tile, start, top, *, static, block_size, score_kind, keep_second_moment):
    """Pass 1 for one rotation block: scores, moments and the running top pose."""
    rotations_block = _block_rows(arrays.rotations, start, block_size)
    proj = _score_window_projection(arrays, rotations_block, static)
    prior = full_row_pose_log_prior(
        tile.coarse_mask,
        _block_rows(arrays.rotation_parent, start, block_size),
        arrays.translation_parent,
        _block_rows(arrays.rotation_log_prior, start, block_size),
        arrays.translation_log_prior,
    )
    full = _SCORE_FUNCTIONS[score_kind](tile.Y1, proj, tile.ctf2, tile.y_norm, prior)
    block_score, block_rotation, block_translation = top_p_from_score_block(
        full.score, rotation_offset=start, candidate_count=1
    )
    # Strict comparison keeps the earlier block on exact ties: its rotation ids
    # are lower, which is the host merge's (score, rotation, translation) order.
    better = block_score[:, 0] > top.score
    top = _TopPose(
        center=jnp.maximum(top.center, jnp.max(full.score, axis=(1, 2))),
        score=jnp.where(better, block_score[:, 0], top.score),
        rotation=jnp.where(better, block_rotation[:, 0], top.rotation),
        translation=jnp.where(better, block_translation[:, 0], top.translation),
    )
    return full.score, full.alpha, (full.G_tri if keep_second_moment else None), top


@jax.jit
def _add_block_partition(partition, score, center):
    return partition + jnp.sum(jnp.exp(score - center[:, None, None]), axis=(1, 2))


@partial(jax.jit, static_argnames=("static", "block_size"), donate_argnums=(0,))
def _backproject_block(carry, arrays, tile, score, alpha, G_tri, center, centered_logZ, start, *, static, block_size):
    """Pass 2 for one rotation block: residual, moments and diagnostics.

    ``proj`` is recomputed from the same deterministic projection as pass 1
    instead of retaining it across the tile.
    """
    rotations_block = _block_rows(arrays.rotations, start, block_size)
    proj = _score_window_projection(arrays, rotations_block, static)
    centered = score - center[:, None, None]
    gamma = jnp.exp(centered - centered_logZ[:, None, None])
    rhs_images, lhs_images = pose_moment_images(
        gamma, alpha, G_tri, tile.Y1_recon, tile.ctf2_recon, rhs_dtype=carry.rhs.dtype, lhs_dtype=carry.lhs_tri.dtype
    )
    # The reconstruction operands equal the score operands without the
    # Hermitian weight (full-real observation: one window), so these residual
    # statistics are already divided by that weight.
    residual_images, correction = residual_statistics_from_moment_images(rhs_images, lhs_images, proj)
    embedding = jnp.einsum("btr,btrq->bq", gamma, alpha[..., 1:])
    indices = arrays.score_indices
    # The half-image adjoint supplies conjugate scatters itself.
    residual = batch_adjoint_slice_volume_maybe_windowed(
        residual_images,
        indices,
        rotations_block,
        carry.residual,
        static.image_shape,
        static.volume_shape,
        static.disc_type,
        True,
        True,
        use_window=indices is not None,
        max_r=static.backprojection_max_r,
    )
    nv = jnp.broadcast_to(arrays.coefficient_noise, (carry.residual_power.size,))
    if indices is None:
        residual_power = carry.residual_power + correction * nv
    else:
        residual_power = carry.residual_power.at[indices].add(correction * nv[indices])
    latent_covariance_trace = _latent_covariance_trace_from_packed_moments(G_tri, alpha, static.basis_size)
    centered_score = centered - centered_logZ[:, None, None]
    rhs, lhs_tri = backproject_moment_images(
        rhs_images,
        lhs_images,
        rotations_block,
        static.image_shape,
        static.volume_shape,
        carry.rhs,
        carry.lhs_tri,
        disc_type_backproject=static.disc_type,
        recon_window_indices=arrays.recon_indices,
        use_recon_window=static.use_recon_window,
        backprojection_max_r=static.backprojection_max_r,
    )
    n_significant = jnp.sum(gamma > 1e-3, axis=(1, 2)).astype(jnp.int32)
    carry = _MomentCarry(
        rhs=rhs,
        lhs_tri=lhs_tri,
        residual=residual,
        residual_power=residual_power,
        embedding=carry.embedding + embedding,
        latent_covariance_trace_sum=carry.latent_covariance_trace_sum + jnp.sum(gamma * latent_covariance_trace),
        pose_entropy_sum=carry.pose_entropy_sum - jnp.sum(jnp.where(gamma > 0, gamma * centered_score, 0)),
        offset_second_sum=carry.offset_second_sum + jnp.sum(gamma * arrays.shift_squared[None, :, None]),
        n_significant=carry.n_significant + n_significant,
    )
    return carry, jnp.sum(gamma, axis=(0, 1))


@jax.jit
def _add_block_embedding(embedding, score, alpha, center, centered_logZ):
    gamma = jnp.exp(score - center[:, None, None] - centered_logZ[:, None, None])
    return embedding + jnp.einsum("btr,btrq->bq", gamma, alpha[..., 1:])


def prepare_full_row_stream(
    experiment_dataset,
    mu,
    W=None,
    *,
    noise_variance,
    rotations,
    translations,
    rotation_log_prior,
    translation_log_prior,
    rotation_parent,
    translation_parent,
    n_coarse_rotations: int,
    n_coarse_translations: int,
    geometry: GeometryConfig,
    schedule: ScheduleConfig,
    scoring: ScoringConfig,
    disc_type: str = "linear_interp",
) -> FullRowStream:
    """Upload the model, fine grids and priors once for an expectation's tiles."""
    if scoring.image_scale_corrections is not None or scoring.class_log_prior != 0.0:
        raise ValueError("Full-row streaming supports unit image scale and no class prior")
    if scoring.score_with_masked_images or scoring.relion_unit_half_weights or not scoring.full_real_observation:
        # Pass 2 forms residuals from the reconstruction-window moment images,
        # which requires the score operands to be those with Hermitian weights.
        raise ValueError("Residual statistics require unmasked, unit-contrast full-real observations")
    rotations = np.asarray(rotations, dtype=np.float32)
    translations = np.asarray(translations, dtype=np.float32)
    rotation_parent = np.asarray(rotation_parent, dtype=np.int32)
    translation_parent = np.asarray(translation_parent, dtype=np.int32)
    n_rot, n_trans = int(rotations.shape[0]), int(translations.shape[0])
    if rotation_parent.shape != (n_rot,) or translation_parent.shape != (n_trans,):
        raise ValueError("Every fine rotation and translation needs one coarse parent")
    if np.any(rotation_parent < 0) or np.any(rotation_parent >= int(n_coarse_rotations)):
        raise ValueError("Fine rotation parent outside the coarse grid")
    if np.any(translation_parent < 0) or np.any(translation_parent >= int(n_coarse_translations)):
        raise ValueError("Fine translation parent outside the coarse grid")
    resolved = prepare_dense_ppca_dataset_inputs(
        experiment_dataset,
        mu,
        W,
        q=geometry.q,
        volume_domain=geometry.volume_domain,
        current_size=geometry.current_size,
        relion_unit_half_weights=False,
        square_window=scoring.square_window,
        full_real_observation=scoring.full_real_observation,
    )
    same_window = (resolved.score_indices is None and resolved.recon_indices is None) or (
        resolved.score_indices is not None
        and resolved.recon_indices is not None
        and np.array_equal(np.asarray(resolved.score_indices), np.asarray(resolved.recon_indices))
    )
    if not same_window:
        raise ValueError("Full-row residuals require one score and reconstruction window")
    forward_config = ForwardModelConfig.from_dataset(
        experiment_dataset, disc_type=disc_type, process_fn=experiment_dataset.process_images
    )
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(noise_variance, resolved.image_shape).squeeze()
    block_size = int(schedule.rotation_block_size)
    blocks = tuple(
        (jnp.asarray(start, dtype=jnp.int32), min(block_size, n_rot - start)) for start in range(0, n_rot, block_size)
    )
    arrays = _StreamArrays(
        augmented=resolved.augmented_half_volumes,
        rotations=jnp.asarray(rotations),
        rotation_log_prior=jnp.asarray(np.asarray(rotation_log_prior, dtype=np.float32)),
        translation_log_prior=jnp.asarray(np.asarray(translation_log_prior, dtype=np.float32)),
        rotation_parent=jnp.asarray(rotation_parent),
        translation_parent=jnp.asarray(translation_parent),
        score_indices=resolved.score_indices,
        recon_indices=resolved.recon_indices,
        coefficient_noise=noise_variance_half,
        shift_squared=jnp.sum(jnp.asarray(translations, jnp.float32) ** 2, axis=-1),
    )
    static = _StreamStatic(
        image_shape=tuple(resolved.image_shape),
        volume_shape=tuple(resolved.volume_shape),
        disc_type=str(disc_type),
        projection_max_r=resolved.projection_max_r,
        backprojection_max_r=resolved.backprojection_max_r,
        relion_texture_interp=bool(scoring.relion_texture_interp),
        use_recon_window=bool(resolved.use_window),
        basis_size=int(resolved.q) + 1,
    )
    return FullRowStream(
        dataset=experiment_dataset,
        arrays=arrays,
        static=static,
        forward_config=forward_config,
        resolved=resolved,
        noise_variance_half=noise_variance_half,
        translations=translations,
        n_coarse_rotations=int(n_coarse_rotations),
        n_coarse_translations=int(n_coarse_translations),
        image_batch_size=int(schedule.image_batch_size),
        blocks=blocks,
        rotation_block_size=block_size,
    )


def _load_tile(stream: FullRowStream, image_indices, significant_rows, *, collect_observation: bool):
    image_indices = np.asarray(image_indices)
    if len(significant_rows) != image_indices.size:
        raise ValueError("One coarse support row is required per tile image")
    if image_indices.size > stream.image_batch_size:
        raise ValueError("A full-row tile must fit one image batch")
    batches = list(stream.dataset.iter_batches(stream.image_batch_size, indices=image_indices, by_image=False))
    if len(batches) != 1:
        raise RuntimeError("Expected exactly one image batch per full-row tile")
    batch_data, _rots, _trans, ctf_params, _noise, _particle_indices, indices = batches[0]
    batch = prepare_dense_ppca_image_batch(
        stream.dataset,
        stream.resolved,
        stream.forward_config,
        stream.noise_variance_half,
        stream.translations,
        batch_data,
        ctf_params,
        indices,
        collect_observation=collect_observation,
    )
    coarse = coarse_support_mask(significant_rows, stream.n_coarse_rotations, stream.n_coarse_translations)
    tile = _TileArrays(
        coarse_mask=jnp.asarray(coarse),
        Y1=batch.Y1_score,
        ctf2=batch.ctf2_score,
        Y1_recon=batch.Y1_recon,
        ctf2_recon=batch.ctf2_recon,
        y_norm=batch.y_norm,
    )
    return tile, batch.observation_power


def _score_tile(stream: FullRowStream, tile: _TileArrays, *, score_kind: str, keep_second_moment: bool):
    """Pass 1 and the centered partition, retaining block scores and moments on the device."""
    n_images = int(tile.y_norm.shape[0])
    top = _TopPose(
        center=jnp.full((n_images,), -jnp.inf, jnp.float32),
        score=jnp.full((n_images,), -jnp.inf, jnp.float32),
        rotation=jnp.full((n_images,), -1, jnp.int32),
        translation=jnp.full((n_images,), -1, jnp.int32),
    )
    retained = []
    for start, size in stream.blocks:
        score, alpha, G_tri, top = _score_block(
            stream.arrays,
            tile,
            start,
            top,
            static=stream.static,
            block_size=size,
            score_kind=score_kind,
            keep_second_moment=keep_second_moment,
        )
        retained.append((score, alpha, G_tri))
    partition = jnp.zeros((n_images,), jnp.float32)
    for score, _alpha, _G_tri in retained:
        partition = _add_block_partition(partition, score, top.center)
    return retained, top, jnp.log(partition)


def _top_pose_posterior(top_centered_score: np.ndarray, centered_logZ: np.ndarray) -> np.ndarray:
    """Host float64 top-1 posterior, as in ``select_distinct_top_poses``."""
    with np.errstate(over="ignore", invalid="ignore"):
        posterior = np.exp(top_centered_score.astype(np.float64) - centered_logZ.astype(np.float64)[:, None])
    return np.where(np.isfinite(top_centered_score), posterior, 0.0).astype(np.float32)


@full_float32
def accumulate_full_row_tile(
    stream: FullRowStream, image_indices, significant_rows, *, factor_once: bool, enforce_x0: bool = True
) -> AugmentedPPCAStats:
    """Accumulate unregularized PPCA statistics for one full-row image tile.

    ``significant_rows`` holds the packed coarse significant pose ids (or
    ``None``) of each tile image. Returns the same statistics and summary
    diagnostics as the host-mask ``accumulate_dense_ppca_statistics`` call
    with ``collect_residuals=True``.
    """
    tile, observation_power = _load_tile(stream, image_indices, significant_rows, collect_observation=True)
    retained, top, centered_logZ = _score_tile(
        stream, tile, score_kind="factor_once" if factor_once else "blocked", keep_second_moment=True
    )
    arrays, static = stream.arrays, stream.static
    P = static.basis_size
    half_size = int(arrays.augmented.shape[1])
    n_images = int(tile.y_norm.shape[0])
    carry = _MomentCarry(
        rhs=jnp.zeros((P, half_size), dtype=jnp.complex64),
        lhs_tri=jnp.zeros((tri_size(P), half_size), dtype=jnp.float32),
        residual=jnp.zeros((P, half_size), dtype=jnp.complex64),
        residual_power=jnp.zeros(arrays.coefficient_noise.shape, jnp.float32) + observation_power,
        embedding=jnp.zeros((n_images, P - 1), jnp.float32),
        latent_covariance_trace_sum=jnp.float32(0),
        pose_entropy_sum=jnp.float32(0),
        offset_second_sum=jnp.float32(0),
        n_significant=jnp.zeros((n_images,), jnp.int32),
    )
    rotation_mass = []
    for index, (start, size) in enumerate(stream.blocks):
        score, alpha, G_tri = retained[index]
        retained[index] = None  # release the block's moments once consumed
        carry, block_mass = _backproject_block(
            carry, arrays, tile, score, alpha, G_tri, top.center, centered_logZ, start, static=static, block_size=size
        )
        rotation_mass.append(block_mass)
    rhs, lhs_tri = carry.rhs, carry.lhs_tri
    if enforce_x0:
        rhs = _enforce_augmented_x0(rhs, static.volume_shape)
        lhs_tri = _enforce_augmented_x0(lhs_tri.astype(jnp.complex64), static.volume_shape).real.astype(jnp.float32)
    weights = make_half_image_weights(static.image_shape)
    shells = make_shell_indices_half(static.image_shape)
    n_shells = int(np.max(np.asarray(shells))) + 1
    residual_num = jnp.zeros(n_shells, jnp.float32).at[shells].add(weights * carry.residual_power)
    residual_den = jnp.zeros(n_shells, jnp.float32).at[shells].add(weights * n_images)
    logZ = top.center + centered_logZ
    host = jax.device_get(
        {
            "log_likelihood": jnp.sum(logZ),
            "top_centered_score": top.score - top.center,
            "centered_logZ": centered_logZ,
            "rotation": top.rotation,
            "translation": top.translation,
            "n_significant": carry.n_significant,
            "rotation_mass": jnp.concatenate(rotation_mass),
            "latent": carry.latent_covariance_trace_sum,
            "entropy": carry.pose_entropy_sum,
            "offset": carry.offset_second_sum,
        }
    )
    pmax = _top_pose_posterior(host["top_centered_score"][:, None], host["centered_logZ"])[:, 0]
    finite = np.isfinite(host["top_centered_score"])
    original_ids = stream.dataset.original_image_indices_from_local(np.asarray(image_indices))
    diagnostics = {
        "engine": FULL_ROW_ENGINE,
        "rotation_block_size": stream.rotation_block_size,
        "pmax_mean": float(jnp.mean(jnp.asarray(pmax))),
        "nsig_mean": float(jnp.mean(jnp.asarray(host["n_significant"]))),
        "log_likelihood": float(host["log_likelihood"]),
        "logZ_mean": float(host["log_likelihood"]) / n_images,
        "max_posterior_per_image": pmax,
        "n_significant_per_image": host["n_significant"],
        "best_rotation_idx": np.where(finite, host["rotation"], -1).astype(np.int32),
        "best_translation_idx": np.where(finite, host["translation"], -1).astype(np.int32),
        "offset_second_sum_px2": float(host["offset"]),
        "rotation_mass": host["rotation_mass"],
        "latent_covariance_trace_mean": float(host["latent"] / np.float32(n_images)),
        "pose_entropy_mean": float(host["entropy"] / np.float32(n_images)),
    }
    return AugmentedPPCAStats(
        rhs=jnp.swapaxes(rhs, 0, 1),
        lhs_tri=jnp.swapaxes(lhs_tri, 0, 1),
        log_likelihood=float(host["log_likelihood"]),
        n_images=n_images,
        residual_gradient=carry.residual.T,
        residual_num=residual_num,
        residual_den=residual_den,
        embeddings=carry.embedding,
        original_image_ids=original_ids,
        diagnostics=diagnostics,
    )


@full_float32
def full_row_tile_embeddings(stream: FullRowStream, image_indices, significant_rows) -> DensePPCAEmbeddings:
    """Pose-marginal embeddings of one full-row tile, as ``compute_dense_ppca_embeddings``."""
    tile, _ = _load_tile(stream, image_indices, significant_rows, collect_observation=False)
    retained, top, centered_logZ = _score_tile(stream, tile, score_kind="blocked", keep_second_moment=False)
    embedding = jnp.zeros((int(tile.y_norm.shape[0]), stream.static.basis_size - 1), jnp.float32)
    for index in range(len(retained)):
        score, alpha, _G_tri = retained[index]
        retained[index] = None
        embedding = _add_block_embedding(embedding, score, alpha, top.center, centered_logZ)
    image_indices = np.asarray(image_indices, dtype=np.int64)
    original_ids = stream.dataset.original_image_indices_from_local(image_indices)
    return DensePPCAEmbeddings(embedding, original_ids, int(image_indices.size))
