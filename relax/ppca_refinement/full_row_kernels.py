"""The device kernels of the streamed PPCA engine and the records they read.

The stream's records (``FullRowStream`` with its static fields and device arrays, a tile's ``TileArrays``, the
kept pass-1 operands and the pass-2 carry), the GEMM precision rule (``resolve_gemm_precision``, ``gemm``), the
pass-1 block scorer (``score_block``, its latent solve and CUDA stages) with the tile's posterior
(``score_tile``), and the pass-2 moment blocks (``moment_block``, ``scatter_moment_images``) with the compacted
pass 2 that skips rows of negligible mass (``run_skipping_pass2``). The driver (``full_row_stream``) and the
tile planner (``full_row_tile_plan``) call them; this module imports neither. The formulation is section 14 of
``docs/math/vdam_ppca_algorithm.md``.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from recovar import core
from recovar.core.configs import ForwardModelConfig
from recovar.ppca.triangular import tri_size

from relax.cuda.ppca import (
    ppca_latent_epilogue_f32,
    ppca_moment_scatter_f32,
    ppca_posterior_prep_f32,
    ppca_window_project_f32,
)
from relax.ppca_refinement.engine import compensated_add
from relax.ppca_refinement.residual_statistics import residual_statistics_from_moment_images
from relax.projection.adjoint import batch_adjoint_slice_volume_maybe_windowed

HIGHEST = jax.lax.Precision.HIGHEST


GEMM_PRECISIONS = ("auto", "fp32", "tf32")


def tf32_capable(device) -> bool:
    """TF32 tensor cores: a GPU of compute capability 8.0 or later."""
    return device.platform == "gpu" and float(getattr(device, "compute_capability", "0")) >= 8.0


def resolve_gemm_precision(requested: str, device) -> str:
    """The stream GEMM precision used on ``device``: ``"auto"`` is tf32 where TF32 tensor cores exist, else fp32."""
    if requested not in GEMM_PRECISIONS:
        raise ValueError(f"gemm_precision must be one of {GEMM_PRECISIONS}")
    if requested == "auto":
        return "tf32" if tf32_capable(device) else "fp32"
    if requested == "tf32" and not tf32_capable(device):
        raise ValueError("TF32 stream GEMMs need a GPU of compute capability 8.0 or later")
    return requested


def gemm(static):
    """Precision of the four stream GEMMs: full float32, or one TF32 tensor-core pass with float32 accumulation."""
    return jax.lax.DotAlgorithmPreset.TF32_TF32_F32 if static.gemm_precision == "tf32" else HIGHEST


def _metric_dot(sums, ctf2, static):
    """The LHS metric GEMM ``sums @ ctf2``: packed moment rows ``sums`` (per row an image's PSD
    ``sum gamma E[[1, z][1, z]^T]``) against the non-negative ``ctf2``.

    TF32 rounds each packed moment entry on its own, which leaves a nearly singular moment matrix
    indefinite by about 2^-11 of its largest entry; summed over images, such rows reached the
    coupled-direction check (VDAM past update 2000 on a realistic fixture). Rounding ``ctf2``
    scales all of a pixel's moment channels by one non-negative number and keeps the sum PSD.
    So under tf32 the moments are split into a TF32 value and the exact float32 remainder,
    and both go through the TF32 GEMM: the moments then carry about 2^-22 relative error and
    the GEMM takes two TF32 passes. It contracts over images only (the RHS and score GEMMs also
    over translations), so the second pass is small.
    """
    if static.gemm_precision != "tf32":
        return jnp.dot(sums, ctf2, precision=gemm(static))
    # Clearing the 13 low mantissa bits leaves a TF32 value (exact in the GEMM) and an exact remainder.
    high = jax.lax.bitcast_convert_type(
        jax.lax.bitcast_convert_type(sums, jnp.uint32) & jnp.uint32(0xFFFFE000), jnp.float32
    )
    split = jnp.dot(jnp.concatenate([high, sums - high]), ctf2, precision=gemm(static))
    return split[: sums.shape[0]] + split[sums.shape[0] :]


# (image, translation) columns per multiple on GPU streams: B T = 4350 at 10076 leaves the TF32
# GEMMs on align1 tensor-core kernels, 1.7x slower on A100 than at 4352 (proto3/gemm_tf32_align.py).
SHIFT_ALIGN = 8


class StreamStatic(NamedTuple):
    """Hashable shape and kernel choices shared by every block program."""

    image_shape: tuple[int, int]
    volume_shape: tuple[int, int, int]
    disc_type: str
    projection_max_r: float | None
    backprojection_max_r: float | None
    relion_texture_interp: bool
    basis_size: int
    metric_trace_only: bool  # backproject sum_p LHS_pp instead of every packed LHS channel
    # GPU streams run relax's fused CUDA stages (windowed projection, latent epilogue, posterior
    # weights, residual scatter); the XLA formulation runs elsewhere and is the float64 reference.
    cuda_kernels: bool
    gemm_precision: str  # resolved: "fp32" (exact float32 products) or "tf32" (TF32 tensor cores, sm_80+)


class StreamArrays(NamedTuple):
    """Device operands fixed for one expectation (model, grids, priors)."""

    augmented: jax.Array  # (P, half_volume) complex64 [mu, W_1..W_q]
    augmented_voxel_major: jax.Array | None  # (half_volume, P) for the CUDA projector; None off GPU
    rotations: jax.Array  # (R + 1, 3, 3) float32 fine rotation grid and a sentinel row
    rotation_log_prior: jax.Array  # (R + 1,) float32
    translation_log_prior: jax.Array  # (T,) float32
    rotation_parent: jax.Array  # (R + 1,) int32 coarse rotation of each fine row; sentinel -> R_coarse
    translation_parent: jax.Array  # (T,) int32 coarse translation of each fine shift
    score_indices: jax.Array | None  # score window in the packed half image
    # CUDA path: the score window (every pixel without one) padded with -1 to a multiple of
    # _GEMM_ALIGN pixels, so the GEMM operands' rows are 16-byte aligned. None off GPU.
    gemm_window: jax.Array | None
    coefficient_noise: jax.Array  # (n_half,) noise variance per half-image pixel
    shift_squared: jax.Array  # (T,) squared fine shift length in px^2


class TileArrays(NamedTuple):
    """Device operands for one image tile, uploaded once per tile.

    A subtomogram tile (section 16.4 of ``docs/math/vdam_ppca_algorithm.md``) has ``K`` tilt
    frames shared by its particles: the pixel axis of every operand is ``(frame, pixel)``, frame
    major, and ``frames`` holds the frame matrices ``Aproj_f``. A single-particle tile has
    ``frames=None`` and ``K = 1``.
    """

    coarse_mask: jax.Array  # (B, R_coarse + 1, T_coarse) bool significant coarse poses; last row False
    rows: jax.Array  # (capacity,) int32 sorted union of supported fine rows, padded with the sentinel R
    Y1: jax.Array  # (K 2F, B * T) float32 [Re; Im] of the weighted shifted score images, per frame (GPU: B T padded)
    ctf2: jax.Array  # (K F, B) float32 score CTF^2 / noise
    Y1_recon: jax.Array  # (B * T, K 2F) float32 [Re, Im] of the reconstruction images, per frame (GPU: B T padded)
    ctf2_recon: jax.Array  # (B, K F) float32
    y_norm: jax.Array  # (B,)
    frames: jax.Array | None = None  # (K, 3, 3) float32 tilt frame matrices; None for single particles


class Posterior(NamedTuple):
    """Tile normalization from the kept pass-1 scores."""

    center: jax.Array  # (B,) score maximum
    centered_logZ: jax.Array  # (B,) log sum exp(score - center)
    top_score: jax.Array  # (B,) first maximum, as center
    top_rotation: jax.Array  # (B,) fine row of the first maximum
    top_translation: jax.Array  # (B,)


class _Kept(NamedTuple):
    """Pass-1 results of every tile pose, rotation-major over the row table."""

    score: jax.Array  # (capacity, B, T) pose log-scores without the image energy; -inf unsupported
    latent_mean: jax.Array  # (q, capacity, B, T) posterior latent means
    latent_covariance: jax.Array  # (capacity, B, tri(q)) packed upper (I + H_zz)^-1
    # Per (row, image) over translations, written by the CUDA epilogue: maximum score, first
    # maximizing translation and sum of exp(score - maximum). The XLA path leaves them unused.
    part_max: jax.Array  # (capacity, B)
    part_arg: jax.Array  # (capacity, B) int32
    part_sum: jax.Array  # (capacity, B)


class _MomentCarry(NamedTuple):
    """Pass-2 accumulators, updated in block order.

    Each rotation block backprojects into zero volumes that are then added to
    the tile sums with Kahan compensation. A single float32 atomic accumulator
    over a whole tile rounds away the many small posterior-tail contributions
    once voxels grow large, a downward bias that grows with images per tile.
    """

    # (groups, half, 32) voxel-major moment volumes: the metric channels (packed LHS, or its
    # trace under metric_trace_only), then the real and imaginary parts of each residual channel.
    moments: jax.Array
    moments_compensation: jax.Array
    residual_power: jax.Array
    embedding: jax.Array  # (B, q)
    rotation_mass: jax.Array  # (capacity,) per row-table position
    latent_covariance_trace_sum: jax.Array
    pose_entropy_sum: jax.Array
    offset_second_sum: jax.Array
    n_significant: jax.Array  # (B,)


class FullRowStream(NamedTuple):
    """Prepared inputs of one streamed full-row expectation over image tiles."""

    dataset: object
    arrays: StreamArrays
    static: StreamStatic
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
    # Reads one tile: ``(stream, image_indices, significant_rows, collect_observation=...)`` ->
    # ``(TileArrays, observation_power, layout)``; ``None`` is the single-particle reader
    # (:func:`_load_tile`). Subtomogram particles read their tilt images
    # (:func:`relax.ppca_initial_model.tomo.load_tilt_tile`).
    tile_loader: object = None
    # Pass 2 skips every pose row on which each tile image's posterior mass is below this floor
    # (0 visits every scored row); :func:`_pass2_rows`.
    pass2_mass_floor: float = 0.0
    # Planned images per tile (:func:`plan_tile_images`), set by the controller; a padding reader rounds
    # each tile up to its :func:`tile_size_bucket`. None: tiles keep their own sizes.
    tile_images: int | None = None


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


def project(arrays: StreamArrays, rotations, static: StreamStatic):
    """Windowed augmented projections ``(P, R, F)`` of the given rotations, component-major."""
    proj = _project_half(arrays, rotations, static)
    return proj if arrays.score_indices is None else proj[:, :, arrays.score_indices]


def _project_half(arrays: StreamArrays, rotations, static: StreamStatic):
    """Augmented projections ``(P, R, n_half)`` on the whole half image."""
    kwargs = {} if static.projection_max_r is None else {"max_r": static.projection_max_r}
    return core.batch_slice_volume(
        arrays.augmented,
        rotations,
        static.image_shape,
        static.volume_shape,
        static.disc_type,
        half_volume=True,
        half_image=True,
        relion_texture_interp=static.relion_texture_interp,
        **kwargs,
    )


def real_imag(values):
    """``[Re, Im]`` along the last axis: real dot products of these are ``Re(conj(a) b)``."""
    return jnp.concatenate([values.real, values.imag], axis=-1)


def _latent_block(Y1, ctf2, proj, pose_log_prior, n_images: int):
    """Pose scores, latent means and covariances for one rotation block.

    ``Y1`` is ``(K 2F, B T)`` (real and imaginary parts per frame), ``ctf2``
    ``(K F, B)``, ``proj`` ``(P, R, K, F)`` and ``pose_log_prior`` ``(B, R, T)``;
    the frame sum is part of each GEMM's contraction (``K = 1`` for single particles). One GEMM gives
    ``Re<Y1, A_p>`` for every pose and component (``t_mx`` for ``p = 0``,
    ``g_zx`` otherwise), a second the CTF/noise-weighted ``Re(conj(A_i) A_j)``
    (``nu_mm``, ``h_zm``, ``H_zz``), as in
    :func:`relax.ppca_refinement.engine._per_pose_stats_block`. With the
    Cholesky factor ``L`` of ``I + H_zz`` and ``b = g_zx - h_zm`` (section 4 of
    the algorithm document) the score is
    ``-0.5 [nu_mm - 2 t_mx - |L^-1 b|^2 + log det(I + H_zz)]`` plus the prior,
    the latent mean ``L^-T L^-1 b`` and the covariance ``L^-T L^-1``. Returns
    score ``(R, B, T)``, means ``(q, R, B, T)`` and packed covariances
    ``(R, B, tri(q))``.
    """
    P, R = proj.shape[:2]
    B = n_images
    T = Y1.shape[1] // B
    inner = jnp.dot(real_imag(proj).reshape(P * R, -1), Y1, precision=HIGHEST).reshape(P, R, B, T)
    first, second = np.triu_indices(P)
    # Static component slices: an index-array gather here lowers to a strided gather fusion.
    real, imag = proj.real, proj.imag
    products = jnp.stack([real[i] * real[j] + imag[i] * imag[j] for i, j in zip(first.tolist(), second.tolist())])
    gram = jnp.dot(products.reshape(first.size * R, -1), ctf2, precision=HIGHEST).reshape(first.size, R, B)
    return latent_scores(inner, gram, pose_log_prior)


def latent_scores(inner, gram, pose_log_prior):
    """:func:`_latent_block` from its two GEMM outputs: ``inner`` ``(P, R, B, T)`` and the packed upper
    ``gram`` ``(tri(P), R, B)``."""
    P, R, B, T = inner.shape
    q = P - 1
    first, second = np.triu_indices(P)
    index = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(first, second))}
    rho = gram[index[(0, 0)], :, :, None] - 2.0 * inner[0]
    prior = jnp.transpose(pose_log_prior, (1, 0, 2))
    if q == 0:
        return -0.5 * rho + prior, jnp.zeros((0, R, B, T), inner.dtype), jnp.zeros((R, B, 0), inner.dtype)
    # The (rotation, image) factor is materialized once; fusing it into every
    # translation's substitution would repeat the factorization per pose.
    H = [[gram[index[(min(i, j), max(i, j))]] for j in range(1, P)] for i in range(1, P)]
    L, logdet = jax.lax.optimization_barrier(_unit_shift_cholesky(H))
    Lt = [[x if x is None else x[..., None] for x in row] for row in L]
    b = [inner[j] - gram[index[(0, j)], :, :, None] for j in range(1, P)]
    # Materialize each substitution stage: a fused consumer would otherwise
    # recompute the whole chain for every latent component it writes.
    v = jax.lax.optimization_barrier(_forward_substitute(Lt, b))
    score = -0.5 * (rho - sum(x * x for x in v) + logdet[..., None]) + prior
    mean = jnp.stack(_back_substitute(Lt, v), axis=0)
    inverse = _lower_inverse(L)
    covariance = jnp.stack(
        [sum(inverse[k][i] * inverse[k][j] for k in range(j, q)) for i, j in zip(*np.triu_indices(q))], axis=-1
    )
    return score, mean, covariance


def _unit_shift_cholesky(H):
    """Lower Cholesky factor of ``I + H`` for symmetric ``H`` given as nested lists of batch arrays.

    Unrolled over the latent rank. Returns the factor (nested lists, ``None``
    above the diagonal) and ``log det(I + H)``.
    """
    q = len(H)
    L = [[None] * q for _ in range(q)]
    log_diagonal = []
    for j in range(q):
        s = 1.0 + H[j][j]
        for k in range(j):
            s = s - L[j][k] * L[j][k]
        L[j][j] = jnp.sqrt(s)
        log_diagonal.append(jnp.log(L[j][j]))
        for i in range(j + 1, q):
            s = H[i][j]
            for k in range(j):
                s = s - L[i][k] * L[j][k]
            L[i][j] = s / L[j][j]
        # Materialize column j: later columns read it many times, and XLA would
        # otherwise duplicate its whole dependency chain into each reader.
        column = jax.lax.optimization_barrier([L[i][j] for i in range(j, q)])
        for i in range(j, q):
            L[i][j] = column[i - j]
    return L, 2.0 * sum(log_diagonal)


def _forward_substitute(L, b):
    """Entries of ``L^-1 b`` for the entries ``b`` of a vector."""
    v = []
    for i in range(len(L)):
        s = b[i]
        for k in range(i):
            s = s - L[i][k] * v[k]
        v.append(s / L[i][i])
    return v


def _back_substitute(L, v):
    """Entries of ``L^-T v``."""
    q = len(L)
    z = [None] * q
    for i in reversed(range(q)):
        s = v[i]
        for k in range(i + 1, q):
            s = s - L[k][i] * z[k]
        z[i] = s / L[i][i]
    return z


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
        column = jax.lax.optimization_barrier([inverse[i][j] for i in range(j, q)])
        for i in range(j, q):
            inverse[i][j] = column[i - j]
    return inverse


def _block_rows(tile, start, size: int):
    return jax.lax.dynamic_slice_in_dim(tile.rows, start, size, axis=0)


def n_frames(tile) -> int:
    return 1 if tile.frames is None else int(tile.frames.shape[0])


def frame_rotations(tile, rotations):
    """Projection rotations ``Aproj_f R_r`` of a block's rows, ``(R K, 3, 3)`` with the frame minor.

    Single-particle tiles project at the rows' own rotations.
    """
    if tile.frames is None:
        return rotations
    expanded = jnp.einsum("fab,rbc->rfac", tile.frames, rotations, precision=HIGHEST)
    return expanded.reshape(-1, 3, 3)


@partial(jax.jit, static_argnames=("static", "block_size"), donate_argnums=(2,))
def score_block(arrays, tile, kept, start, *, static, block_size):
    """Pass 1 for one rotation block, written into the kept tile results."""
    rows = _block_rows(tile, start, block_size)
    if static.cuda_kernels:
        return _score_block_cuda(arrays, tile, kept, start, rows, static)
    prior = full_row_pose_log_prior(
        tile.coarse_mask,
        arrays.rotation_parent[rows],
        arrays.translation_parent,
        arrays.rotation_log_prior[rows],
        arrays.translation_log_prior,
    )
    proj = project(arrays, frame_rotations(tile, arrays.rotations[rows]), static)
    proj = proj.reshape(proj.shape[0], rows.shape[0], n_frames(tile), proj.shape[-1])
    score, mean, covariance = _latent_block(tile.Y1, tile.ctf2, proj, prior, tile.y_norm.shape[0])
    return kept._replace(
        score=jax.lax.dynamic_update_slice_in_dim(kept.score, score, start, axis=0),
        latent_mean=jax.lax.dynamic_update_slice_in_dim(kept.latent_mean, mean, start, axis=1),
        latent_covariance=jax.lax.dynamic_update_slice_in_dim(kept.latent_covariance, covariance, start, axis=0),
    )


def score_window_pixels(arrays, static):
    """Score-window pixels of the packed half image (every pixel without a window)."""
    if arrays.score_indices is not None:
        return arrays.score_indices
    return jnp.arange(int(static.image_shape[0]) * (int(static.image_shape[1]) // 2 + 1), dtype=jnp.int32)


def _score_block_cuda(arrays, tile, kept, start, rows, static):
    """:func:`_latent_block` with the projections and its elementwise stages in relax's CUDA kernels.

    The projector writes the planar ``[Re | Im]`` score operand and the pair products of the
    Gram GEMM; the epilogue writes the scores, latent moments and the per-row normalization
    partials into the kept buffers in place. The pose log-prior is evaluated per pose from the
    same tables as :func:`full_row_pose_log_prior`.
    """
    P = static.basis_size
    planar, products = ppca_window_project_f32(
        arrays.augmented_voxel_major,
        arrays.gemm_window,
        frame_rotations(tile, arrays.rotations[rows]),
        image_shape=static.image_shape,
        volume_shape=static.volume_shape,
        max_r=static.projection_max_r,
        with_products=True,
    )
    # Rows are (rotation, frame) with the frame minor, so each rotation's frames are one GEMM row.
    R = rows.shape[0]
    # (P, R, B T padded): the epilogue reads each row's B T columns and ignores the padding.
    inner = jnp.dot(planar.reshape(P * R, -1), tile.Y1, precision=gemm(static)).reshape(P, R, -1)
    gram = jnp.dot(products.reshape(products.shape[0] * R, -1), tile.ctf2, precision=gemm(static)).reshape(
        -1, R, tile.ctf2.shape[1]
    )
    tables = (
        arrays.rotation_parent,
        arrays.rotation_log_prior,
        arrays.translation_parent,
        arrays.translation_log_prior,
        tile.coarse_mask,
    )
    return _Kept(*ppca_latent_epilogue_f32(inner, gram, rows, tables, start, kept))


@partial(jax.jit, static_argnames=("n_blocks", "block_size"))
def _normalize(score, rows, *, n_blocks, block_size):
    """Maxima, centered log-partitions and first-maximum poses from the kept scores.

    The top pose is the first maximum in block order, then translation, then
    row within the block (the row table ascends), as the block-by-block merge.
    """
    _, B, T = score.shape
    scored = score[: n_blocks * block_size]
    center = jnp.max(scored, axis=(0, 2))
    centered_logZ = jnp.log(jnp.sum(jnp.exp(scored - center[None, :, None]), axis=(0, 2)))
    ordered = jnp.transpose(scored.reshape(n_blocks, block_size, B, T), (2, 0, 3, 1)).reshape(B, -1)
    flat = jnp.argmax(ordered, axis=1)
    block, rest = flat // (T * block_size), flat % (T * block_size)
    position = block * block_size + rest % block_size
    return Posterior(
        center=center,
        centered_logZ=centered_logZ,
        top_score=jnp.max(ordered, axis=1),
        top_rotation=rows[position],
        top_translation=(rest // block_size).astype(jnp.int32),
    )


@partial(jax.jit, static_argnames=("n_blocks", "block_size"))
def _normalize_partials(kept, rows, *, n_blocks, block_size):
    """:func:`_normalize` from the CUDA epilogue's per-(row, image) partials.

    The partition is ``sum_r part_sum[r] exp(part_max[r] - center)``. Among the rows whose
    maximum is the image maximum, the top pose is the smallest (block, translation, row) key,
    the first maximum in :func:`_normalize`'s order.
    """
    n = n_blocks * block_size
    part_max, part_arg, part_sum = kept.part_max[:n], kept.part_arg[:n], kept.part_sum[:n]
    T = kept.score.shape[2]
    center = jnp.max(part_max, axis=0)
    centered_logZ = jnp.log(jnp.sum(part_sum * jnp.exp(part_max - center[None]), axis=0))
    position = jnp.arange(n, dtype=jnp.int32)[:, None]
    key = (position // block_size) * (T * block_size) + part_arg * block_size + position % block_size
    first = jnp.min(jnp.where(part_max == center[None], key, jnp.iinfo(jnp.int32).max), axis=0)
    block, rest = first // (T * block_size), first % (T * block_size)
    return Posterior(
        center=center,
        centered_logZ=centered_logZ,
        top_score=center,
        top_rotation=rows[block * block_size + rest % block_size],
        top_translation=(rest // block_size).astype(jnp.int32),
    )


def second_moment_sums(gamma, mean, covariance):
    """``sum_t gamma E[[1, z][1, z]^T]`` packed upper ``(tri(P), R, B)`` per rotation and image."""
    q = mean.shape[0]
    cov_index = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(*np.triu_indices(q)))}
    entries = []
    for i, j in zip(*np.triu_indices(q + 1)):
        if i == 0:
            entries.append(gamma if j == 0 else gamma * mean[j - 1])
        else:
            second = covariance[..., cov_index[(i - 1, j - 1)], None] + mean[i - 1] * mean[j - 1]
            entries.append(gamma * second)
    return jnp.stack([jnp.sum(x, axis=-1) for x in entries], axis=0)


@partial(jax.jit, static_argnames=("static", "block_size", "moments"), donate_argnums=(0,))
def moment_block(carry, arrays, tile, kept, posterior, start, *, static, block_size, moments):
    """Pass 2 for one rotation block: posterior weights, M-step images, residuals and diagnostics."""
    if static.cuda_kernels:
        weights, sums, carry = _posterior_block_cuda(
            carry, arrays, kept, posterior, start, static, block_size, tile.Y1_recon.shape[0]
        )
    else:
        weights, sums, carry = _posterior_block(carry, arrays, kept, posterior, start, block_size)
    if not moments:
        return carry
    P = static.basis_size
    # One image per (rotation, frame), frame minor: the GEMM outputs' rows split into frames.
    R = block_size * n_frames(tile)
    F = tile.ctf2_recon.shape[1] // n_frames(tile)
    n_moments = sums.shape[0]  # packed (1 + q) x (1 + q) moment channels
    rotations = frame_rotations(tile, arrays.rotations[_block_rows(tile, start, block_size)])
    rhs_parts = jnp.dot(weights.reshape(P * block_size, -1), tile.Y1_recon, precision=gemm(static)).reshape(
        P, R, 2 * F
    )
    lhs_images = _metric_dot(sums.reshape(n_moments * block_size, -1), tile.ctf2_recon, static).reshape(n_moments, R, F)
    return scatter_moment_images(carry, arrays, static, rhs_parts, lhs_images, rotations)


def scatter_moment_images(carry, arrays, static, rhs_parts, lhs_images, rotations):
    """Backproject one block's M-step images into the carry: the packed LHS metric images ``(tri(P), R, F)``
    and the planar RHS images ``(P, R, 2F)`` at ``rotations`` ``(R, 3, 3)``, as residual images, and their
    noise-power correction."""
    P = static.basis_size
    F = lhs_images.shape[2]
    # The reconstruction operands equal the score operands without the
    # Hermitian weight (full-real observation: one window), so these residual
    # statistics are already divided by that weight. The RHS images enter the
    # residual only: neither optimizer reads an RHS volume, so only the LHS
    # metric and the residual gradient are backprojected.
    indices = arrays.score_indices
    if static.cuda_kernels:
        window = arrays.gemm_window
        projections, _ = ppca_window_project_f32(
            arrays.augmented_voxel_major,
            window,
            rotations,
            image_shape=static.image_shape,
            volume_shape=static.volume_shape,
            max_r=static.projection_max_r,
            with_products=False,
        )
        block, correction = ppca_moment_scatter_f32(
            lhs_images,
            rhs_parts,
            projections,
            window,
            rotations,
            image_shape=static.image_shape,
            volume_shape=static.volume_shape,
            max_r=static.backprojection_max_r,
            metric_trace=static.metric_trace_only,
        )
        correction = correction[: score_window_pixels(arrays, static).shape[0]]  # padding pixels carry none
    else:
        window = score_window_pixels(arrays, static)
        proj = project(arrays, rotations, static)
        rhs_images = jax.lax.complex(rhs_parts[..., :F], rhs_parts[..., F:])  # (P, R, F)
        residual_images, correction = residual_statistics_from_moment_images(rhs_images, lhs_images, proj)
        if static.metric_trace_only:
            # The adjoint is linear per channel, so the trace of the LHS volume is the
            # adjoint of the per-pose trace images.
            tri_i, tri_j = np.triu_indices(P)
            lhs_images = jnp.sum(lhs_images[np.flatnonzero(tri_i == tri_j)], axis=0, keepdims=True)
        block = _backproject_moments(lhs_images, residual_images, window, rotations, static)
    nv = jnp.broadcast_to(arrays.coefficient_noise, (carry.residual_power.size,))
    if indices is None:
        residual_power = carry.residual_power + correction * nv
    else:
        residual_power = carry.residual_power.at[indices].add(correction * nv[indices])
    moments, moments_compensation = compensated_add(carry.moments, carry.moments_compensation, block)
    return carry._replace(moments=moments, moments_compensation=moments_compensation, residual_power=residual_power)


def _posterior_block(carry, arrays, kept, posterior, start, block_size):
    """Posterior weights ``(P, R, B, T)``, translation moment sums ``(tri(P), R, B)`` and diagnostics."""
    score = jax.lax.dynamic_slice_in_dim(kept.score, start, block_size, axis=0)
    mean = jax.lax.dynamic_slice_in_dim(kept.latent_mean, start, block_size, axis=1)
    covariance = jax.lax.dynamic_slice_in_dim(kept.latent_covariance, start, block_size, axis=0)
    centered_score = (score - posterior.center[None, :, None]) - posterior.centered_logZ[None, :, None]
    gamma = jnp.exp(centered_score)  # (R, B, T)
    diagonal = [k for k, (i, j) in enumerate(zip(*np.triu_indices(mean.shape[0]))) if i == j]
    latent_trace = sum((covariance[..., k] for k in diagonal), jnp.zeros(covariance.shape[:2], covariance.dtype))
    carry = carry._replace(
        embedding=carry.embedding + jnp.einsum("rbt,qrbt->bq", gamma, mean, precision=HIGHEST),
        rotation_mass=jax.lax.dynamic_update_slice_in_dim(carry.rotation_mass, jnp.sum(gamma, axis=(1, 2)), start, 0),
        latent_covariance_trace_sum=carry.latent_covariance_trace_sum + jnp.sum(gamma * latent_trace[..., None]),
        pose_entropy_sum=carry.pose_entropy_sum - jnp.sum(jnp.where(gamma > 0, gamma * centered_score, 0)),
        offset_second_sum=carry.offset_second_sum + jnp.sum(gamma * arrays.shift_squared),
        n_significant=carry.n_significant + jnp.sum(gamma > 1e-3, axis=(0, 2)).astype(jnp.int32),
    )
    weights = jnp.concatenate([gamma[None], gamma[None] * mean], axis=0)
    return weights, second_moment_sums(gamma, mean, covariance), carry


def _posterior_block_cuda(carry, arrays, kept, posterior, start, static, block_size, row_length):
    """:func:`_posterior_block` with the per-pose stage in relax's CUDA kernel.

    The kernel sums over translations per (rotation, image); the remaining reductions over
    rotations and images are small. ``sums`` entry ``(0, 0)`` is the posterior mass and entries
    ``(0, j)`` the latent-mean sums, so the embedding, rotation mass and latent trace come from them.
    """
    weights, sums, partial, count = ppca_posterior_prep_f32(
        kept.score,
        kept.latent_mean,
        kept.latent_covariance,
        posterior.center,
        posterior.centered_logZ,
        arrays.shift_squared,
        start,
        block_size=block_size,
        row_length=row_length,
    )
    q = static.basis_size - 1
    covariance = jax.lax.dynamic_slice_in_dim(kept.latent_covariance, start, block_size, axis=0)
    latent_trace = sum(covariance[..., k] for k, (i, j) in enumerate(zip(*np.triu_indices(q))) if i == j)
    mass = sums[0]  # (R, B)
    carry = carry._replace(
        embedding=carry.embedding + jnp.sum(sums[1 : q + 1], axis=1).T,
        rotation_mass=jax.lax.dynamic_update_slice_in_dim(carry.rotation_mass, jnp.sum(mass, axis=1), start, 0),
        latent_covariance_trace_sum=carry.latent_covariance_trace_sum + jnp.sum(mass * latent_trace),
        pose_entropy_sum=carry.pose_entropy_sum + jnp.sum(partial[0]),
        offset_second_sum=carry.offset_second_sum + jnp.sum(partial[1]),
        n_significant=carry.n_significant + jnp.sum(count, axis=0),
    )
    return weights, sums, carry


def moment_groups(static):
    """Groups of 32 voxel-major channels: the metric channels and the residual's real/imaginary parts."""
    P = static.basis_size
    metric = 1 if static.metric_trace_only else tri_size(P)
    return -(-(metric + 2 * P) // 32)


def unpack_moments(moments, static):
    """Metric ``(channels, half)`` and residual gradient ``(half, P)`` of ``(groups, half, 32)`` moment volumes."""
    P = static.basis_size
    metric = 1 if static.metric_trace_only else tri_size(P)
    flat = jnp.transpose(moments, (1, 0, 2)).reshape(moments.shape[1], -1)
    residual = flat[:, metric : metric + 2 * P].reshape(-1, P, 2)
    return flat[:, :metric].T, jax.lax.complex(residual[..., 0], residual[..., 1])


def _backproject_moments(real_images, complex_images, indices, rotations, static):
    """``(groups, half, 32)`` linear-interpolation adjoint of one block's windowed half-image moment images.

    Real channels ``(C_r, R, F)`` then the real and imaginary parts of the complex channels
    ``(C_c, R, F)``, into zero half volumes, with the shared windowed adjoint per channel type:
    the layout :func:`relax.cuda.ppca.ppca_moment_scatter_f32` produces on GPU streams.
    """
    half = int(np.prod(static.volume_shape[:2])) * (int(static.volume_shape[2]) // 2 + 1)

    def adjoint(images, dtype):
        return batch_adjoint_slice_volume_maybe_windowed(
            images,
            indices,
            rotations,
            jnp.zeros((images.shape[0], half), dtype),
            static.image_shape,
            static.volume_shape,
            static.disc_type,
            True,
            True,
            use_window=True,
            max_r=static.backprojection_max_r,
        )

    real = adjoint(real_images, real_images.dtype)
    complex_volume = adjoint(complex_images, complex_images.dtype)
    channels = [real.T, jnp.stack([complex_volume.real.T, complex_volume.imag.T], axis=-1).reshape(half, -1)]
    flat = jnp.concatenate(channels, axis=1)
    groups = -(-flat.shape[1] // 32)
    flat = jnp.pad(flat, ((0, 0), (0, groups * 32 - flat.shape[1])))
    return jnp.transpose(flat.reshape(half, groups, 32), (1, 0, 2))


def score_tile(stream: FullRowStream, tile: TileArrays, n_blocks: int, kept: _Kept | None = None):
    """Pass 1 and the tile normalization; returns the kept results and the posterior summary.

    ``kept`` may be the previous tile's buffer of the same shape: pass 1 overwrites
    every row of the first ``n_blocks`` blocks, and nothing reads the rows after them.
    """
    n_images = int(tile.y_norm.shape[0])
    T = int(stream.translations.shape[0])
    q = stream.static.basis_size - 1
    capacity = len(stream.block_starts) * stream.rotation_block_size
    if kept is None or kept.score.shape != (capacity, n_images, T):
        kept = empty_kept(capacity, n_images, T, q, jnp.float32)
    for start in stream.block_starts[:n_blocks]:
        kept = score_block(
            stream.arrays, tile, kept, start, static=stream.static, block_size=stream.rotation_block_size
        )
    return kept, _tile_posterior(kept, tile.rows, n_blocks, stream.rotation_block_size, stream.static)


def empty_kept(capacity: int, n_images: int, n_translations: int, q: int, dtype) -> _Kept:
    return _Kept(
        score=jnp.full((capacity, n_images, n_translations), -jnp.inf, dtype),
        latent_mean=jnp.zeros((q, capacity, n_images, n_translations), dtype),
        latent_covariance=jnp.zeros((capacity, n_images, tri_size(q)), dtype),
        part_max=jnp.full((capacity, n_images), -jnp.inf, dtype),
        part_arg=jnp.zeros((capacity, n_images), jnp.int32),
        part_sum=jnp.zeros((capacity, n_images), dtype),
    )


def _tile_posterior(kept: _Kept, rows, n_blocks: int, block_size: int, static: StreamStatic) -> Posterior:
    if static.cuda_kernels:
        return _normalize_partials(kept, rows, n_blocks=n_blocks, block_size=block_size)
    return _normalize(kept.score, rows, n_blocks=n_blocks, block_size=block_size)


def _run_pass2(stream, tile, kept, posterior, n_blocks, carry, *, moments):
    for start in stream.block_starts[:n_blocks]:
        carry = moment_block(
            carry,
            stream.arrays,
            tile,
            kept,
            posterior,
            start,
            static=stream.static,
            block_size=stream.rotation_block_size,
            moments=moments,
        )
    return carry


def empty_carry(stream, n_images, observation_power):
    arrays, static = stream.arrays, stream.static
    P = static.basis_size
    half_size = int(arrays.augmented.shape[1])
    capacity = len(stream.block_starts) * stream.rotation_block_size
    moments = (moment_groups(static), half_size, 32)
    return _MomentCarry(
        moments=jnp.zeros(moments, dtype=jnp.float32),
        moments_compensation=jnp.zeros(moments, dtype=jnp.float32),
        residual_power=jnp.zeros(arrays.coefficient_noise.shape, jnp.float32) + observation_power,
        embedding=jnp.zeros((n_images, P - 1), jnp.float32),
        rotation_mass=jnp.zeros((capacity,), jnp.float32),
        latent_covariance_trace_sum=jnp.float32(0),
        pose_entropy_sum=jnp.float32(0),
        offset_second_sum=jnp.float32(0),
        n_significant=jnp.zeros((n_images,), jnp.int32),
    )


@partial(jax.jit, static_argnames=("n",))
def _row_mass(kept, posterior, *, n):
    """Largest posterior mass of each of the first ``n`` pose rows over the tile images."""
    log_norm = posterior.center + posterior.centered_logZ
    mass = kept.part_sum[:n] * jnp.exp(kept.part_max[:n] - log_norm[None, :])
    return jnp.max(mass, axis=1)


@partial(jax.jit, static_argnames=("n",))
def _row_mass_xla(score, posterior, *, n):
    log_norm = posterior.center + posterior.centered_logZ
    return jnp.max(jnp.sum(jnp.exp(score[:n] - log_norm[None, :, None]), axis=2), axis=1)


# Pass 2 compacts its rows only when it keeps at most this share of them; above it, the gather
# would cost about what it saves, so pass 2 visits every row in place (no mass is dropped).
PASS2_COMPACT_MAX_FRACTION = 0.5


def _pass2_rows(stream, kept, posterior, n_blocks):
    """Positions, in pass-1 row order, of the rows pass 2 visits; ``None`` when it visits all.

    A row is skipped when every tile image's posterior mass on it (summed over translations) is
    below ``stream.pass2_mass_floor``, so each image loses less than ``rows * floor`` of its mass.
    When more than :data:`PASS2_COMPACT_MAX_FRACTION` of the rows remain, none is skipped.
    """
    if stream.pass2_mass_floor <= 0:
        return None
    n = n_blocks * stream.rotation_block_size
    if stream.static.cuda_kernels:
        mass = _row_mass(kept, posterior, n=n)
    else:
        mass = _row_mass_xla(kept.score, posterior, n=n)
    positions = np.flatnonzero(np.asarray(jax.device_get(mass)) >= stream.pass2_mass_floor)
    return None if positions.size > PASS2_COMPACT_MAX_FRACTION * n else positions.astype(np.int32)


@partial(jax.jit, static_argnames=("capacity",))
def _compact_rows(kept, rows, positions, count, sentinel, *, capacity):
    """Pass-1 results and row table at ``positions[:count]``, padded with masked sentinel rows."""
    valid = jnp.arange(capacity) < count
    index = jnp.where(valid, positions, 0)
    take = partial(jnp.take, indices=index, axis=0)
    compact = _Kept(
        score=jnp.where(valid[:, None, None], take(kept.score), -jnp.inf),
        latent_mean=jnp.take(kept.latent_mean, index, axis=1),
        latent_covariance=take(kept.latent_covariance),
        part_max=jnp.where(valid[:, None], take(kept.part_max), -jnp.inf),
        part_arg=take(kept.part_arg),
        part_sum=jnp.where(valid[:, None], take(kept.part_sum), 0.0),
    )
    table = jnp.full(rows.shape, sentinel, rows.dtype).at[:capacity].set(jnp.where(valid, take(rows), sentinel))
    return compact, table


def _compact_capacity(total_blocks: int, n_blocks: int) -> int:
    """Blocks of the compacted pass-2 buffer: the smallest of ``total / 2^k`` (k = 1..5, rounded up) that
    holds ``n_blocks``. Compaction keeps at most half the rows, so the buffer is at most half the kept
    one (:func:`stream_tile_bytes` counts it), and a stage compiles at most five buffer shapes."""
    sizes = sorted({max(1, -(-total_blocks // (1 << k))) for k in range(1, 6)})
    return next((size for size in sizes if size >= n_blocks), total_blocks)


@jax.jit
def _expand_rotation_mass(compact_mass, positions, count):
    """Rotation masses of the compacted rows back at their pass-1 positions (padding adds zeros)."""
    valid = jnp.arange(positions.shape[0]) < count
    values = jnp.where(valid, compact_mass[: positions.shape[0]], 0.0)
    return jnp.zeros_like(compact_mass).at[positions].add(values)


def run_skipping_pass2(stream, tile, kept, posterior, layout, carry, *, moments):
    """Pass 2 over the rows :func:`_pass2_rows` keeps; per-row outputs return to pass-1 positions."""
    positions = _pass2_rows(stream, kept, posterior, layout["n_blocks"])
    layout["pass2_rows"] = layout["n_blocks"] * stream.rotation_block_size if positions is None else int(positions.size)
    if positions is None:
        return _run_pass2(stream, tile, kept, posterior, layout["n_blocks"], carry, moments=moments)
    count = int(positions.size)
    n_blocks = max(1, -(-count // stream.rotation_block_size))
    capacity = _compact_capacity(len(stream.block_starts), n_blocks) * stream.rotation_block_size
    padded = np.zeros(capacity, np.int32)
    padded[:count] = positions
    sentinel = np.int32(stream.rotation_parent.size)
    compact, table = _compact_rows(kept, tile.rows, jnp.asarray(padded), np.int32(count), sentinel, capacity=capacity)
    carry = _run_pass2(stream, tile._replace(rows=table), compact, posterior, n_blocks, carry, moments=moments)
    return carry._replace(
        rotation_mass=_expand_rotation_mass(carry.rotation_mass, jnp.asarray(padded), np.int32(count))
    )
