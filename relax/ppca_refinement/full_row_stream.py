"""Device-resident PPCA statistics for streamed rows of the shared fine grid.

Fine orientations are the children of coarse orientations and do not depend on
the image, so every image of a tile can score rows of one shared fine grid.
A tile scores the sorted union of its images' supported rows; each image's
exact support (the coarse pass-1 significant set expanded to fine children) is
a device-expanded pose log-prior that is ``-inf`` outside it, including on the
union rows of other images. When every image keeps every coarse orientation the
union is the full grid. The tile's coarse support and row table are uploaded
once; the row table has a fixed capacity padded with a masked sentinel row, so
blocks keep one shape and varying support never recompiles.

Per image tile:

1. Pass 1 scores every pose of every rotation block with one real GEMM of the
   shifted images against the augmented projections and one of the CTF/noise
   weights against the projection products (:func:`_pose_scores`), and keeps
   the scores of the whole tile on the device (:func:`_score_block`).
2. The tile's maxima, centered log-partitions, top poses and the
   ``(image, row)`` pairs with any nonzero float32 posterior weight follow
   from the kept scores (:func:`_normalize`).
3. Pass 2 visits, per rotation block, only the images and rows of those
   pairs (:func:`_moment_block`): it recomputes the latent moments there and
   forms the M-step images, expected residuals and diagnostics. Every pose it
   skips has weight exactly zero, so it would only add zeros; the statistics
   are those of the full posterior.

The formulation is section 14 of ``docs/math/vdam_ppca_algorithm.md``.
Tests compare the engine against the host-mask reference
:func:`relax.ppca_refinement.dense_dataset.accumulate_dense_ppca_statistics`.
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
    _project_augmented_half_volumes,
    prepare_dense_ppca_dataset_inputs,
    prepare_dense_ppca_image_batch,
)
from relax.ppca_refinement.engine import (
    _enforce_augmented_x0,
    backproject_moment_images,
    compensated_add,
    pose_invariant_score_offset,
)
from relax.ppca_refinement.residual_statistics import full_float32, residual_statistics_from_moment_images

FULL_ROW_ENGINE = "full_row_device_resident"

_HIGHEST = jax.lax.Precision.HIGHEST


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
    rotations: jax.Array  # (R + 1, 3, 3) float32 fine rotation grid and a sentinel row
    rotation_log_prior: jax.Array  # (R + 1,) float32
    translation_log_prior: jax.Array  # (T,) float32
    rotation_parent: jax.Array  # (R + 1,) int32 coarse rotation of each fine row; sentinel -> R_coarse
    translation_parent: jax.Array  # (T,) int32 coarse translation of each fine shift
    score_indices: jax.Array | None  # score window in the packed half image
    recon_indices: jax.Array | None  # reconstruction window
    coefficient_noise: jax.Array  # (n_half,) noise variance per half-image pixel
    shift_squared: jax.Array  # (T,) squared fine shift length in px^2


class _TileArrays(NamedTuple):
    """Device operands for one image tile, uploaded once per tile."""

    coarse_mask: jax.Array  # (B, R_coarse + 1, T_coarse) bool significant coarse poses; last row False
    rows: jax.Array  # (capacity,) int32 sorted union of supported fine rows, padded with the sentinel R
    Y1: jax.Array  # (B, T, 2F) float32 [Re, Im] of the weighted shifted score images
    ctf2: jax.Array  # (B, F) float32
    Y1_recon: jax.Array  # (B, T, 2F) float32 [Re, Im] of the reconstruction images
    ctf2_recon: jax.Array  # (B, F) float32
    y_norm: jax.Array  # (B,)


class _Posterior(NamedTuple):
    """Tile normalization from the kept pass-1 scores."""

    center: jax.Array  # (B,) score maximum
    centered_logZ: jax.Array  # (B,) log sum exp(score - center)
    top_score: jax.Array  # (B,) first maximum, as center
    top_rotation: jax.Array  # (B,) fine row of the first maximum
    top_translation: jax.Array  # (B,)
    active: jax.Array  # (B, capacity) bool: some translation has a nonzero weight


class _MomentCarry(NamedTuple):
    """Pass-2 accumulators, updated in block order.

    Each rotation block backprojects into zero volumes that are then added to
    the tile sums with Kahan compensation. A single float32 atomic accumulator
    over a whole tile rounds away the many small posterior-tail contributions
    once voxels grow large, a downward bias that grows with images per tile.
    """

    rhs: jax.Array
    lhs_tri: jax.Array
    residual: jax.Array
    rhs_compensation: jax.Array
    lhs_compensation: jax.Array
    residual_compensation: jax.Array
    residual_power: jax.Array
    embedding: jax.Array  # (B + 1, q); the last row absorbs padded images
    rotation_mass: jax.Array  # (capacity + 1,); the last entry absorbs padded rows
    latent_covariance_trace_sum: jax.Array
    pose_entropy_sum: jax.Array
    offset_second_sum: jax.Array
    n_significant: jax.Array  # (B + 1,)


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
    block_starts: tuple[jax.Array, ...]  # device start of each row-table block
    rotation_block_size: int
    rotation_parent: np.ndarray  # (R,) host copy for the per-tile row union
    device: object  # the jax device holding this stream's operands and running its tiles


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


def tile_fine_rows(coarse_mask: np.ndarray, rotation_parent: np.ndarray) -> np.ndarray:
    """Sorted fine rows whose coarse parent is significant for any tile image."""
    return np.flatnonzero(np.any(coarse_mask, axis=(0, 2))[rotation_parent]).astype(np.int32)


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


def _project(arrays: _StreamArrays, rotations, static: _StreamStatic):
    """Windowed augmented projections ``(R, P, F)`` of the given rotations."""
    proj = _project_augmented_half_volumes(
        arrays.augmented,
        rotations,
        static.image_shape,
        static.volume_shape,
        static.disc_type,
        max_r=static.projection_max_r,
        relion_texture_interp=static.relion_texture_interp,
    )
    return proj if arrays.score_indices is None else proj[:, :, arrays.score_indices]


def _real_imag(values):
    """``[Re, Im]`` along the last axis: real dot products of these are ``Re(conj(a) b)``."""
    return jnp.concatenate([values.real, values.imag], axis=-1)


def _unit_shift_cholesky(H):
    """Lower Cholesky factor of ``I + H`` for symmetric ``H`` (..., q, q), unrolled over ``q``.

    Returns the factor as nested lists of batch arrays and ``log det(I + H)``.
    """
    q = H.shape[-1]
    L = [[None] * q for _ in range(q)]
    log_diagonal = []
    for j in range(q):
        s = 1.0 + H[..., j, j]
        for k in range(j):
            s = s - L[j][k] * L[j][k]
        L[j][j] = jnp.sqrt(s)
        log_diagonal.append(jnp.log(L[j][j]))
        for i in range(j + 1, q):
            s = H[..., i, j]
            for k in range(j):
                s = s - L[i][k] * L[j][k]
            L[i][j] = s / L[j][j]
    return L, 2.0 * sum(log_diagonal)


def _forward_substitute(L, b):
    """``L^-1 b`` for ``b`` (..., q); ``L`` entries broadcast against ``b[..., 0]``."""
    v = []
    for i in range(len(L)):
        s = b[..., i]
        for k in range(i):
            s = s - L[i][k] * v[k]
        v.append(s / L[i][i])
    return v


def _lower_inverse(L):
    """Entries of ``L^-1`` for a lower-triangular nested-list factor."""
    q = len(L)
    inverse = [[None] * q for _ in range(q)]
    for j in range(q):
        inverse[j][j] = 1.0 / L[j][j]
        for i in range(j + 1, q):
            s = 0.0
            for k in range(j, i):
                s = s + L[i][k] * inverse[k][j]
            inverse[i][j] = -s / L[i][i]
    return inverse


def _latent_system(Y1, ctf2, proj):
    """Pose-score building blocks of :func:`relax.ppca_refinement.engine._per_pose_stats_block`.

    ``Y1`` is ``(B, T, 2F)`` (real and imaginary parts), ``ctf2`` ``(B, F)`` and
    ``proj`` ``(R, P, F)``. One GEMM gives ``Re<Y1, A_p>`` for every pose and
    component: ``t_mx`` (``p = 0``) and ``g_zx``. A second gives the
    CTF/noise-weighted products ``Re(conj(A_i) A_j)``: ``nu_mm``, ``h_zm`` and
    ``H_zz``. Returns ``(t_mx (B,T,R), nu_mm (B,R), b = g_zx - h_zm (B,T,R,q),
    H_zz (B,R,q,q))``.
    """
    B, T, F2 = Y1.shape
    R, P, F = proj.shape
    q = P - 1
    inner = jnp.dot(Y1.reshape(B * T, F2), _real_imag(proj).reshape(R * P, F2).T, precision=_HIGHEST)
    inner = inner.reshape(B, T, R, P)
    first, second = np.triu_indices(P)
    products = proj.real[:, first] * proj.real[:, second] + proj.imag[:, first] * proj.imag[:, second]
    gram = jnp.dot(ctf2, products.reshape(R * first.size, F).T, precision=_HIGHEST).reshape(B, R, first.size)
    index = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(first, second))}
    nu_mm = gram[..., index[(0, 0)]]
    if q == 0:
        return inner[..., 0], nu_mm, inner[..., 1:], jnp.zeros((B, R, 0, 0), inner.dtype)
    h_zm = jnp.stack([gram[..., index[(0, j)]] for j in range(1, P)], axis=-1)
    H_zz = jnp.stack(
        [jnp.stack([gram[..., index[(min(i, j), max(i, j))]] for j in range(1, P)], axis=-1) for i in range(1, P)],
        axis=-2,
    )
    return inner[..., 0], nu_mm, inner[..., 1:] - h_zm[:, None], H_zz


def _pose_scores(Y1, ctf2, proj, pose_log_prior):
    """Pose log-scores ``(B, T, R)`` without the pose-invariant image energy.

    ``-0.5 [-2 t_mx + nu_mm - b^T (I + H_zz)^-1 b + log det(I + H_zz)]`` plus the
    pose log-prior ``(B, R, T)``; ``b^T (I + H)^-1 b = |L^-1 b|^2`` with the
    Cholesky factor ``L`` of ``I + H`` (section 4 of the algorithm document).
    """
    t_mx, nu_mm, b, H = _latent_system(Y1, ctf2, proj)
    rho = nu_mm[:, None, :] - 2.0 * t_mx
    if b.shape[-1] == 0:
        score = -0.5 * rho
    else:
        L, logdet = _unit_shift_cholesky(H)
        v = _forward_substitute([[x if x is None else x[:, None] for x in row] for row in L], b)
        score = -0.5 * (rho - sum(x * x for x in v) + logdet[:, None])
    return score + jnp.swapaxes(pose_log_prior, -1, -2)


def _back_substitute(L, v):
    """``L^-T v`` for the nested-list lower factor ``L`` and the entries ``v`` of ``L^-1 b``."""
    q = len(L)
    z = [None] * q
    for i in reversed(range(q)):
        s = v[i]
        for k in range(i + 1, q):
            s = s - L[k][i] * z[k]
        z[i] = s / L[i][i]
    return z


def _latent_moments(Y1, ctf2, proj):
    """Posterior latent means ``z`` (B,T,R,q) and covariance ``(I + H)^-1`` (B,R,q,q).

    ``z = L^-T L^-1 b`` by forward and back substitution with the Cholesky
    factor ``L`` of ``I + H``; the covariance is ``L^-T L^-1``.
    """
    _, _, b, H = _latent_system(Y1, ctf2, proj)
    q = b.shape[-1]
    if q == 0:
        return b, H
    L, _ = _unit_shift_cholesky(H)
    Lb = [[x if x is None else x[:, None] for x in row] for row in L]
    z = jnp.stack(_back_substitute(Lb, _forward_substitute(Lb, b)), axis=-1)
    inverse = _lower_inverse(L)
    covariance = jnp.stack(
        [jnp.stack([sum(inverse[k][i] * inverse[k][j] for k in range(max(i, j), q)) for j in range(q)], axis=-1)
         for i in range(q)],
        axis=-2,
    )
    return z, covariance


def _block_rows(tile, start, size: int):
    return jax.lax.dynamic_slice_in_dim(tile.rows, start, size, axis=0)


@partial(jax.jit, static_argnames=("static", "block_size"), donate_argnums=(2,))
def _score_block(arrays, tile, kept, start, *, static, block_size):
    """Pass 1 for one rotation block: scores of every tile pose, written into ``kept``."""
    rows = _block_rows(tile, start, block_size)
    prior = full_row_pose_log_prior(
        tile.coarse_mask,
        arrays.rotation_parent[rows],
        arrays.translation_parent,
        arrays.rotation_log_prior[rows],
        arrays.translation_log_prior,
    )
    score = _pose_scores(tile.Y1, tile.ctf2, _project(arrays, arrays.rotations[rows], static), prior)
    return jax.lax.dynamic_update_slice_in_dim(kept, score, start, axis=2)


@partial(jax.jit, static_argnames=("n_blocks", "block_size"))
def _normalize(kept, rows, *, n_blocks, block_size):
    """Maxima, centered log-partitions, first-maximum poses and nonzero-weight pairs.

    The top pose is the first maximum in block order, then translation, then
    row within the block (the row table ascends), as the block-by-block merge.
    """
    B, T, _ = kept.shape
    scored = kept[:, :, : n_blocks * block_size]
    center = jnp.max(scored, axis=(1, 2))
    centered = scored - center[:, None, None]
    centered_logZ = jnp.log(jnp.sum(jnp.exp(centered), axis=(1, 2)))
    gamma = jnp.exp(centered - centered_logZ[:, None, None])
    active = jnp.any(gamma > 0, axis=1)
    ordered = jnp.swapaxes(scored.reshape(B, T, n_blocks, block_size), 1, 2).reshape(B, -1)
    flat = jnp.argmax(ordered, axis=1)
    block, rest = flat // (T * block_size), flat % (T * block_size)
    position = block * block_size + rest % block_size
    pad = jnp.zeros((B, kept.shape[2] - n_blocks * block_size), bool)
    return _Posterior(
        center=center,
        centered_logZ=centered_logZ,
        top_score=jnp.max(ordered, axis=1),
        top_rotation=rows[position],
        top_translation=(rest // block_size).astype(jnp.int32),
        active=jnp.concatenate([active, pad], axis=1),
    )


def _pack_moments(z, covariance):
    """Packed upper triangle ``(..., tri(P))`` of ``E[[1, z][1, z]^T]`` per pose."""
    q = z.shape[-1]
    P = q + 1
    entries = []
    for i, j in zip(*np.triu_indices(P)):
        if i == 0:
            entries.append(jnp.ones_like(z[..., 0]) if j == 0 else z[..., j - 1])
        else:
            entries.append(covariance[:, None, :, i - 1, j - 1] + z[..., i - 1] * z[..., j - 1])
    return jnp.stack(entries, axis=-1)


@partial(jax.jit, static_argnames=("static", "moments"), donate_argnums=(0,))
def _moment_block(carry, arrays, tile, kept, posterior, images, image_valid, positions, row_valid, *, static, moments):
    """Pass 2 over ``images x positions`` of one rotation block.

    ``images`` (n_b,) and ``positions`` (n_r,) list the block's images and
    row-table positions with a nonzero weight, padded with invalid entries whose
    weights are zero. Every other pose of the block has weight exactly zero.
    """
    B = tile.Y1.shape[0]
    rows = tile.rows[positions]
    rotations = arrays.rotations[rows]
    proj = _project(arrays, rotations, static)
    Y1 = tile.Y1[images]
    ctf2 = tile.ctf2[images]
    score = kept[images[:, None, None], jnp.arange(kept.shape[1])[None, :, None], positions[None, None, :]]
    centered = score - posterior.center[images][:, None, None]
    centered_score = centered - posterior.centered_logZ[images][:, None, None]
    valid = image_valid[:, None, None] & row_valid[None, None, :]
    gamma = jnp.where(valid, jnp.exp(centered_score), 0.0)  # (n_b, T, n_r)
    z, covariance = _latent_moments(Y1, ctf2, proj)
    n_b, T, n_r = gamma.shape
    image_slot = jnp.where(image_valid, images, B)
    position_slot = jnp.where(row_valid, positions, carry.rotation_mass.shape[0] - 1)
    latent_trace = jnp.trace(covariance, axis1=-2, axis2=-1)  # (n_b, n_r)
    carry = carry._replace(
        embedding=carry.embedding.at[image_slot].add(jnp.einsum("btr,btrq->bq", gamma, z, precision=_HIGHEST)),
        rotation_mass=carry.rotation_mass.at[position_slot].add(jnp.sum(gamma, axis=(0, 1))),
        latent_covariance_trace_sum=carry.latent_covariance_trace_sum + jnp.sum(gamma * latent_trace[:, None, :]),
        pose_entropy_sum=carry.pose_entropy_sum - jnp.sum(jnp.where(gamma > 0, gamma * centered_score, 0)),
        offset_second_sum=carry.offset_second_sum + jnp.sum(gamma * arrays.shift_squared[None, :, None]),
        n_significant=carry.n_significant.at[image_slot].add(jnp.sum(gamma > 1e-3, axis=(1, 2)).astype(jnp.int32)),
    )
    if not moments:
        return carry
    P = z.shape[-1] + 1
    F = ctf2.shape[-1]
    alpha = jnp.concatenate([jnp.ones_like(z[..., :1]), z], axis=-1)  # (n_b, T, n_r, P)
    weights = (gamma[..., None] * alpha).transpose(2, 3, 0, 1).reshape(n_r * P, n_b * T)
    rhs_parts = jnp.dot(weights, tile.Y1_recon[images].reshape(n_b * T, 2 * F), precision=_HIGHEST)
    rhs_parts = rhs_parts.reshape(n_r, P, 2 * F)
    rhs_images = jnp.swapaxes(jax.lax.complex(rhs_parts[..., :F], rhs_parts[..., F:]), 0, 1)  # (P, n_r, F)
    G_sum = jnp.einsum("btr,btrk->rkb", gamma, _pack_moments(z, covariance), precision=_HIGHEST)
    K = G_sum.shape[1]
    lhs_images = jnp.dot(G_sum.reshape(n_r * K, n_b), tile.ctf2_recon[images], precision=_HIGHEST)
    lhs_images = jnp.swapaxes(lhs_images.reshape(n_r, K, F), 0, 1)  # (K, n_r, F)
    # The reconstruction operands equal the score operands without the
    # Hermitian weight (full-real observation: one window), so these residual
    # statistics are already divided by that weight.
    residual_images, correction = residual_statistics_from_moment_images(rhs_images, lhs_images, proj)
    indices = arrays.score_indices
    # The half-image adjoint supplies conjugate scatters itself.
    residual_block = batch_adjoint_slice_volume_maybe_windowed(
        residual_images,
        indices,
        rotations,
        jnp.zeros_like(carry.residual),
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
    rhs_block, lhs_block = backproject_moment_images(
        rhs_images,
        lhs_images,
        rotations,
        static.image_shape,
        static.volume_shape,
        jnp.zeros_like(carry.rhs),
        jnp.zeros_like(carry.lhs_tri),
        disc_type_backproject=static.disc_type,
        recon_window_indices=arrays.recon_indices,
        use_recon_window=static.use_recon_window,
        backprojection_max_r=static.backprojection_max_r,
    )
    rhs, rhs_compensation = compensated_add(carry.rhs, carry.rhs_compensation, rhs_block)
    lhs_tri, lhs_compensation = compensated_add(carry.lhs_tri, carry.lhs_compensation, lhs_block)
    residual, residual_compensation = compensated_add(carry.residual, carry.residual_compensation, residual_block)
    return carry._replace(
        rhs=rhs,
        lhs_tri=lhs_tri,
        residual=residual,
        rhs_compensation=rhs_compensation,
        lhs_compensation=lhs_compensation,
        residual_compensation=residual_compensation,
        residual_power=residual_power,
    )


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
    device=None,
) -> FullRowStream:
    """Upload the model, fine grids and priors once to ``device`` for an expectation's tiles."""
    device = jax.local_devices()[0] if device is None else device
    with jax.default_device(device):
        return _prepare_full_row_stream(
            experiment_dataset,
            mu,
            W,
            noise_variance=noise_variance,
            rotations=rotations,
            translations=translations,
            rotation_log_prior=rotation_log_prior,
            translation_log_prior=translation_log_prior,
            rotation_parent=rotation_parent,
            translation_parent=translation_parent,
            n_coarse_rotations=n_coarse_rotations,
            n_coarse_translations=n_coarse_translations,
            geometry=geometry,
            schedule=schedule,
            scoring=scoring,
            disc_type=disc_type,
            device=device,
        )


def _prepare_full_row_stream(
    experiment_dataset,
    mu,
    W,
    *,
    noise_variance,
    rotations,
    translations,
    rotation_log_prior,
    translation_log_prior,
    rotation_parent,
    translation_parent,
    n_coarse_rotations,
    n_coarse_translations,
    geometry,
    schedule,
    scoring,
    disc_type,
    device,
) -> FullRowStream:
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
    block_starts = tuple(jnp.asarray(start, dtype=jnp.int32) for start in range(0, n_rot, block_size))
    # The sentinel row (identity rotation, zero prior) has the all-unsupported
    # coarse parent R_coarse: it pads the last block of every tile.
    sentinel_rotation = np.eye(3, dtype=np.float32)[None]
    arrays = _StreamArrays(
        augmented=resolved.augmented_half_volumes,
        rotations=jnp.asarray(np.concatenate([rotations, sentinel_rotation])),
        rotation_log_prior=jnp.asarray(np.append(np.asarray(rotation_log_prior, dtype=np.float32), np.float32(0))),
        translation_log_prior=jnp.asarray(np.asarray(translation_log_prior, dtype=np.float32)),
        rotation_parent=jnp.asarray(np.append(rotation_parent, np.int32(n_coarse_rotations))),
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
        block_starts=block_starts,
        rotation_block_size=block_size,
        rotation_parent=rotation_parent,
        device=device,
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
    batch_data, ctf_params = jax.device_put((batch_data, ctf_params), stream.device)
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
    rows = tile_fine_rows(coarse, stream.rotation_parent)
    n_blocks = -(-rows.size // stream.rotation_block_size)
    capacity = len(stream.block_starts) * stream.rotation_block_size
    table = np.full(capacity, stream.rotation_parent.size, dtype=np.int32)
    table[: rows.size] = rows
    unsupported = np.zeros((coarse.shape[0], 1, coarse.shape[2]), dtype=bool)
    tile = _TileArrays(
        coarse_mask=jnp.asarray(np.concatenate([coarse, unsupported], axis=1)),
        rows=jnp.asarray(table),
        Y1=_real_imag(batch.Y1_score),
        ctf2=batch.ctf2_score,
        Y1_recon=_real_imag(batch.Y1_recon),
        ctf2_recon=batch.ctf2_recon,
        y_norm=batch.y_norm,
    )
    # Exact per-image rows versus the scored union, for the wasted-work record.
    supported = np.any(coarse, axis=2)[:, stream.rotation_parent].sum()
    layout = {
        "rows": rows,
        "n_blocks": n_blocks,
        "scored_rows": int(n_blocks * stream.rotation_block_size),
        "supported_image_rows": int(supported),
    }
    return tile, batch.observation_power, layout


def _bucket(count: int, limit: int) -> int:
    """Padded size of ``count`` entries: a power of four from 16, capped at ``limit``."""
    size = 16
    while size < count:
        size *= 4
    return min(size, limit)


def _pass2_plan(active: np.ndarray, n_blocks: int, block_size: int):
    """Per rotation block, the images and row positions with a nonzero weight, padded to buckets."""
    n_images = active.shape[0]
    plan = []
    for block in range(n_blocks):
        sub = active[:, block * block_size : (block + 1) * block_size]
        images = np.flatnonzero(sub.any(axis=1))
        if images.size == 0:
            continue
        positions = np.flatnonzero(sub.any(axis=0)) + block * block_size
        n_b, n_r = _bucket(images.size, n_images), _bucket(positions.size, block_size)
        image_valid = np.arange(n_b) < images.size
        row_valid = np.arange(n_r) < positions.size
        plan.append(
            (
                np.pad(images, (0, n_b - images.size)).astype(np.int32),
                image_valid,
                np.pad(positions, (0, n_r - positions.size), constant_values=positions[0]).astype(np.int32),
                row_valid,
            )
        )
    return plan


def _score_tile(stream: FullRowStream, tile: _TileArrays, n_blocks: int):
    """Pass 1 and the tile normalization; returns the kept scores and the posterior summary."""
    n_images = int(tile.y_norm.shape[0])
    T = int(stream.translations.shape[0])
    capacity = len(stream.block_starts) * stream.rotation_block_size
    kept = jnp.full((n_images, T, capacity), -jnp.inf, jnp.float32)
    for start in stream.block_starts[:n_blocks]:
        kept = _score_block(
            stream.arrays, tile, kept, start, static=stream.static, block_size=stream.rotation_block_size
        )
    posterior = _normalize(kept, tile.rows, n_blocks=n_blocks, block_size=stream.rotation_block_size)
    finite = jax.device_get((jnp.all(jnp.isfinite(posterior.center)), jnp.all(jnp.isfinite(posterior.centered_logZ))))
    if not all(finite):
        raise ValueError("Every full-row image needs a finite supported pose and partition")
    return kept, posterior


def _run_pass2(stream, tile, kept, posterior, n_blocks, carry, *, moments):
    plan = _pass2_plan(np.asarray(jax.device_get(posterior.active)), n_blocks, stream.rotation_block_size)
    for images, image_valid, positions, row_valid in plan:
        carry = _moment_block(
            carry,
            stream.arrays,
            tile,
            kept,
            posterior,
            jnp.asarray(images),
            jnp.asarray(image_valid),
            jnp.asarray(positions),
            jnp.asarray(row_valid),
            static=stream.static,
            moments=moments,
        )
    return carry, sum(int(r.sum()) * int(i.sum()) for i, _, _, r in plan)


def _empty_carry(stream, n_images, observation_power):
    arrays, static = stream.arrays, stream.static
    P = static.basis_size
    half_size = int(arrays.augmented.shape[1])
    capacity = len(stream.block_starts) * stream.rotation_block_size
    return _MomentCarry(
        rhs=jnp.zeros((P, half_size), dtype=jnp.complex64),
        lhs_tri=jnp.zeros((tri_size(P), half_size), dtype=jnp.float32),
        residual=jnp.zeros((P, half_size), dtype=jnp.complex64),
        rhs_compensation=jnp.zeros((P, half_size), dtype=jnp.complex64),
        lhs_compensation=jnp.zeros((tri_size(P), half_size), dtype=jnp.float32),
        residual_compensation=jnp.zeros((P, half_size), dtype=jnp.complex64),
        residual_power=jnp.zeros(arrays.coefficient_noise.shape, jnp.float32) + observation_power,
        embedding=jnp.zeros((n_images + 1, P - 1), jnp.float32),
        rotation_mass=jnp.zeros((capacity + 1,), jnp.float32),
        latent_covariance_trace_sum=jnp.float32(0),
        pose_entropy_sum=jnp.float32(0),
        offset_second_sum=jnp.float32(0),
        n_significant=jnp.zeros((n_images + 1,), jnp.int32),
    )


def _top_pose_posterior(top_centered_score: np.ndarray, centered_logZ: np.ndarray) -> np.ndarray:
    """Host float64 top-1 posterior, as in ``select_distinct_top_poses``."""
    with np.errstate(over="ignore", invalid="ignore"):
        posterior = np.exp(top_centered_score.astype(np.float64) - centered_logZ.astype(np.float64)[:, None])
    return np.where(np.isfinite(top_centered_score), posterior, 0.0).astype(np.float32)


@full_float32
def accumulate_full_row_tile(
    stream: FullRowStream,
    image_indices,
    significant_rows,
    *,
    enforce_x0: bool = True,
) -> AugmentedPPCAStats:
    """Accumulate unregularized PPCA statistics for one full-row image tile.

    ``significant_rows`` holds the packed coarse significant pose ids (or
    ``None``) of each tile image. Returns the same statistics and summary
    diagnostics as the host-mask ``accumulate_dense_ppca_statistics`` call
    with ``collect_residuals=True``.
    """
    with jax.default_device(stream.device):
        return _accumulate_full_row_tile(stream, image_indices, significant_rows, enforce_x0=enforce_x0)


def _accumulate_full_row_tile(stream, image_indices, significant_rows, *, enforce_x0):
    tile, observation_power, layout = _load_tile(stream, image_indices, significant_rows, collect_observation=True)
    kept, posterior = _score_tile(stream, tile, layout["n_blocks"])
    static = stream.static
    n_images = int(tile.y_norm.shape[0])
    carry = _empty_carry(stream, n_images, observation_power)
    carry, pass2_pairs = _run_pass2(stream, tile, kept, posterior, layout["n_blocks"], carry, moments=True)
    del kept
    rhs, lhs_tri = carry.rhs, carry.lhs_tri
    if enforce_x0:
        rhs = _enforce_augmented_x0(rhs, static.volume_shape)
        lhs_tri = _enforce_augmented_x0(lhs_tri.astype(jnp.complex64), static.volume_shape).real.astype(jnp.float32)
    weights = make_half_image_weights(static.image_shape)
    shells = make_shell_indices_half(static.image_shape)
    n_shells = int(np.max(np.asarray(shells))) + 1
    residual_num = jnp.zeros(n_shells, jnp.float32).at[shells].add(weights * carry.residual_power)
    residual_den = jnp.zeros(n_shells, jnp.float32).at[shells].add(weights * n_images)
    # Scores exclude the pose-invariant image constant; the absolute log-partition adds it back.
    logZ = posterior.center + posterior.centered_logZ + pose_invariant_score_offset(tile.y_norm)
    host = jax.device_get(
        {
            "log_likelihood": jnp.sum(logZ),
            "top_centered_score": posterior.top_score - posterior.center,
            "centered_logZ": posterior.centered_logZ,
            "rotation": posterior.top_rotation,
            "translation": posterior.top_translation,
            "n_significant": carry.n_significant[:n_images],
            "rotation_mass": carry.rotation_mass[:-1],
            "latent": carry.latent_covariance_trace_sum,
            "entropy": carry.pose_entropy_sum,
            "offset": carry.offset_second_sum,
        }
    )
    pmax = _top_pose_posterior(host["top_centered_score"][:, None], host["centered_logZ"])[:, 0]
    # Every fine row appears at most once in the table; sentinel padding is dropped.
    rotation_mass = np.zeros(stream.rotation_parent.size, np.float32)
    rotation_mass[layout["rows"]] = host["rotation_mass"][: layout["rows"].size]
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
        "rotation_mass": rotation_mass,
        "scored_image_rows": n_images * layout["scored_rows"],
        "supported_image_rows": layout["supported_image_rows"],
        "pass2_image_rows": pass2_pairs,
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
        embeddings=carry.embedding[:n_images],
        original_image_ids=original_ids,
        diagnostics=diagnostics,
    )


@full_float32
def full_row_tile_embeddings(stream: FullRowStream, image_indices, significant_rows) -> DensePPCAEmbeddings:
    """Pose-marginal embeddings of one full-row tile, as ``compute_dense_ppca_embeddings``."""
    with jax.default_device(stream.device):
        tile, _, layout = _load_tile(stream, image_indices, significant_rows, collect_observation=False)
        kept, posterior = _score_tile(stream, tile, layout["n_blocks"])
        n_images = int(tile.y_norm.shape[0])
        carry = _empty_carry(stream, n_images, jnp.float32(0))
        carry, _ = _run_pass2(stream, tile, kept, posterior, layout["n_blocks"], carry, moments=False)
        image_indices = np.asarray(image_indices, dtype=np.int64)
        original_ids = stream.dataset.original_image_indices_from_local(image_indices)
        return DensePPCAEmbeddings(carry.embedding[:n_images], original_ids, int(image_indices.size))
