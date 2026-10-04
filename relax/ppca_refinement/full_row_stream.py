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

1. Pass 1, per rotation block (:func:`_score_block`): one real GEMM of the
   augmented projections against the shifted images gives every pose's
   inner products, one of the projection products against the CTF/noise
   weights gives the latent Gram; the pose scores, latent means and latent
   covariances (:func:`_latent_block`) are kept on the device for the tile.
2. The tile's maxima, centered log-partitions and top poses follow from the
   kept scores (:func:`_normalize`).
3. Pass 2, per rotation block (:func:`_moment_block`): the posterior weights
   and the kept moments form the M-step images with two more GEMMs (weighted
   images and weighted CTF/noise), then expected residuals, diagnostics and
   the augmented backprojection. Nothing of pass 1 is recomputed except the
   projections the residuals need.

On GPU streams the elementwise stages around the GEMMs run in relax's CUDA
kernels (:func:`_score_block_cuda`, :func:`_posterior_block_cuda`, the window
projector and the moment scatter); the XLA formulation is the CPU path and the
tests' float64 reference.

Kept operands are component-major ``(component, R, B, T)``, the layout of the
projector's output and of the backprojector's input, so each GEMM reads and
writes them in place and no operand is transposed. Every posterior weight is
used: float32 weights of these posteriors are nonzero for essentially every
(image, rotation) pair.

The formulation is section 14 of ``docs/math/vdam_ppca_algorithm.md``.
Tests compare the engine against the host-mask reference
:func:`relax.ppca_refinement.dense_dataset.accumulate_dense_ppca_statistics`.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from recovar import core
from recovar.core.configs import ForwardModelConfig
from recovar.core.slicing import decide_order
from recovar.ppca.pose_accumulators import AugmentedPPCAStats
from recovar.ppca.triangular import tri_size
from recovar.reconstruction import noise as noise_utils

from relax.cuda.kernels import (
    ppca_latent_epilogue_f32,
    ppca_moment_scatter_f32,
    ppca_posterior_prep_f32,
    ppca_window_project_f32,
)
from relax.helpers.adjoint import batch_adjoint_slice_volume_maybe_windowed
from relax.helpers.half_spectrum import make_half_image_weights, make_shell_indices_half
from relax.ppca_refinement.config import GeometryConfig, ScheduleConfig, ScoringConfig
from relax.ppca_refinement.dense_dataset import (
    DensePPCAEmbeddings,
    prepare_dense_ppca_dataset_inputs,
    prepare_dense_ppca_image_batch,
)
from relax.ppca_refinement.engine import (
    _enforce_augmented_x0,
    compensated_add,
    pose_invariant_score_offset,
)
from relax.ppca_refinement.residual_statistics import full_float32, residual_statistics_from_moment_images

FULL_ROW_ENGINE = "full_row_device_resident"


@dataclasses.dataclass(frozen=True)
class TracePPCAStats(AugmentedPPCAStats):
    """Streamed statistics whose LHS metric is only its per-frequency trace.

    Momentum SGD reads the augmented metric only through ``sum_p LHS_pp``
    (``docs/math/ppca_momentum_sgd.md``), so its streams backproject that one
    channel: ``lhs_tri`` is ``None`` and ``metric_trace`` has shape
    ``(n_frequency,)``.
    """

    metric_trace: jax.Array | None = None


logger = logging.getLogger(__name__)
_HIGHEST = jax.lax.Precision.HIGHEST
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


def _gemm(static):
    """Precision of the four stream GEMMs: full float32, or one TF32 tensor-core pass with float32 accumulation."""
    return jax.lax.DotAlgorithmPreset.TF32_TF32_F32 if static.gemm_precision == "tf32" else _HIGHEST


# Window pixels per GEMM row multiple on GPU streams: unaligned fp32 operands (2F = 3002 at
# 10076) select cuBLAS's align1 kernels, about 4% slower on A100 (jobs/local_*/gemm_align).
_GEMM_ALIGN = 4
# (image, translation) columns per multiple on GPU streams: B T = 4350 at 10076 leaves the TF32
# GEMMs on align1 tensor-core kernels, 1.7x slower on A100 than at 4352 (proto3/gemm_tf32_align.py).
_SHIFT_ALIGN = 8


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
    metric_trace_only: bool  # backproject sum_p LHS_pp instead of every packed LHS channel
    # GPU streams run relax's fused CUDA stages (windowed projection, latent epilogue, posterior
    # weights, residual scatter); the XLA formulation runs elsewhere and is the float64 reference.
    cuda_kernels: bool
    gemm_precision: str  # resolved: "fp32" (exact float32 products) or "tf32" (TF32 tensor cores, sm_80+)


class _StreamArrays(NamedTuple):
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
    recon_indices: jax.Array | None  # reconstruction window
    coefficient_noise: jax.Array  # (n_half,) noise variance per half-image pixel
    shift_squared: jax.Array  # (T,) squared fine shift length in px^2


class _TileArrays(NamedTuple):
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


class _Posterior(NamedTuple):
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
    # Reads one tile: ``(stream, image_indices, significant_rows, collect_observation=...)`` ->
    # ``(_TileArrays, observation_power, layout)``; ``None`` is the single-particle reader
    # (:func:`_load_tile`). Subtomogram particles read their tilt images
    # (:func:`relax.ppca_initial_model.tomo.load_tilt_tile`).
    tile_loader: object = None
    # Pass 2 skips every pose row on which each tile image's posterior mass is below this floor
    # (0 visits every scored row); :func:`_pass2_rows`.
    pass2_mass_floor: float = 0.0
    # Planned images per tile (:func:`plan_tile_images`), set by the controller; a padding reader rounds
    # each tile up to its :func:`tile_size_bucket`. None: tiles keep their own sizes.
    tile_images: int | None = None


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
    """Windowed augmented projections ``(P, R, F)`` of the given rotations, component-major."""
    proj = _project_half(arrays, rotations, static)
    return proj if arrays.score_indices is None else proj[:, :, arrays.score_indices]


def _project_half(arrays: _StreamArrays, rotations, static: _StreamStatic):
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


def _real_imag(values):
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
    inner = jnp.dot(_real_imag(proj).reshape(P * R, -1), Y1, precision=_HIGHEST).reshape(P, R, B, T)
    first, second = np.triu_indices(P)
    # Static component slices: an index-array gather here lowers to a strided gather fusion.
    real, imag = proj.real, proj.imag
    products = jnp.stack([real[i] * real[j] + imag[i] * imag[j] for i, j in zip(first.tolist(), second.tolist())])
    gram = jnp.dot(products.reshape(first.size * R, -1), ctf2, precision=_HIGHEST).reshape(first.size, R, B)
    return _latent_scores(inner, gram, pose_log_prior)


def _latent_scores(inner, gram, pose_log_prior):
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


def _n_frames(tile) -> int:
    return 1 if tile.frames is None else int(tile.frames.shape[0])


def _frame_rotations(tile, rotations):
    """Projection rotations ``Aproj_f R_r`` of a block's rows, ``(R K, 3, 3)`` with the frame minor.

    Single-particle tiles project at the rows' own rotations.
    """
    if tile.frames is None:
        return rotations
    expanded = jnp.einsum("fab,rbc->rfac", tile.frames, rotations, precision=_HIGHEST)
    return expanded.reshape(-1, 3, 3)


@partial(jax.jit, static_argnames=("static", "block_size"), donate_argnums=(2,))
def _score_block(arrays, tile, kept, start, *, static, block_size):
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
    proj = _project(arrays, _frame_rotations(tile, arrays.rotations[rows]), static)
    proj = proj.reshape(proj.shape[0], rows.shape[0], _n_frames(tile), proj.shape[-1])
    score, mean, covariance = _latent_block(tile.Y1, tile.ctf2, proj, prior, tile.y_norm.shape[0])
    return kept._replace(
        score=jax.lax.dynamic_update_slice_in_dim(kept.score, score, start, axis=0),
        latent_mean=jax.lax.dynamic_update_slice_in_dim(kept.latent_mean, mean, start, axis=1),
        latent_covariance=jax.lax.dynamic_update_slice_in_dim(kept.latent_covariance, covariance, start, axis=0),
    )


def _window(arrays, static):
    """Score-window pixels of the packed half image (every pixel without a window)."""
    if arrays.score_indices is not None:
        return arrays.score_indices
    return jnp.arange(int(static.image_shape[0]) * (int(static.image_shape[1]) // 2 + 1), dtype=jnp.int32)


def _gemm_window(score_indices, image_shape) -> jax.Array:
    """The score window padded with -1 (no pixel) to a multiple of :data:`_GEMM_ALIGN`."""
    n_half = int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)
    window = np.arange(n_half, dtype=np.int32) if score_indices is None else np.asarray(score_indices, np.int32)
    pad = -window.size % _GEMM_ALIGN
    return jnp.asarray(np.concatenate([window, np.full(pad, -1, np.int32)]))


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
        _frame_rotations(tile, arrays.rotations[rows]),
        image_shape=static.image_shape,
        volume_shape=static.volume_shape,
        max_r=static.projection_max_r,
        with_products=True,
    )
    # Rows are (rotation, frame) with the frame minor, so each rotation's frames are one GEMM row.
    R = rows.shape[0]
    # (P, R, B T padded): the epilogue reads each row's B T columns and ignores the padding.
    inner = jnp.dot(planar.reshape(P * R, -1), tile.Y1, precision=_gemm(static)).reshape(P, R, -1)
    gram = jnp.dot(products.reshape(products.shape[0] * R, -1), tile.ctf2, precision=_gemm(static)).reshape(
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
    return _Posterior(
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
    return _Posterior(
        center=center,
        centered_logZ=centered_logZ,
        top_score=center,
        top_rotation=rows[block * block_size + rest % block_size],
        top_translation=(rest // block_size).astype(jnp.int32),
    )


def _second_moment_sums(gamma, mean, covariance):
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
def _moment_block(carry, arrays, tile, kept, posterior, start, *, static, block_size, moments):
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
    R = block_size * _n_frames(tile)
    F = tile.ctf2_recon.shape[1] // _n_frames(tile)
    n_moments = sums.shape[0]  # packed (1 + q) x (1 + q) moment channels
    rotations = _frame_rotations(tile, arrays.rotations[_block_rows(tile, start, block_size)])
    rhs_parts = jnp.dot(weights.reshape(P * block_size, -1), tile.Y1_recon, precision=_gemm(static)).reshape(
        P, R, 2 * F
    )
    lhs_images = jnp.dot(sums.reshape(n_moments * block_size, -1), tile.ctf2_recon, precision=_gemm(static)).reshape(
        n_moments, R, F
    )
    return _scatter_moment_images(carry, arrays, static, rhs_parts, lhs_images, rotations)


def _scatter_moment_images(carry, arrays, static, rhs_parts, lhs_images, rotations):
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
        correction = correction[: _window(arrays, static).shape[0]]  # padding pixels carry none
    else:
        window = _window(arrays, static)
        proj = _project(arrays, rotations, static)
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
        embedding=carry.embedding + jnp.einsum("rbt,qrbt->bq", gamma, mean, precision=_HIGHEST),
        rotation_mass=jax.lax.dynamic_update_slice_in_dim(carry.rotation_mass, jnp.sum(gamma, axis=(1, 2)), start, 0),
        latent_covariance_trace_sum=carry.latent_covariance_trace_sum + jnp.sum(gamma * latent_trace[..., None]),
        pose_entropy_sum=carry.pose_entropy_sum - jnp.sum(jnp.where(gamma > 0, gamma * centered_score, 0)),
        offset_second_sum=carry.offset_second_sum + jnp.sum(gamma * arrays.shift_squared),
        n_significant=carry.n_significant + jnp.sum(gamma > 1e-3, axis=(0, 2)).astype(jnp.int32),
    )
    weights = jnp.concatenate([gamma[None], gamma[None] * mean], axis=0)
    return weights, _second_moment_sums(gamma, mean, covariance), carry


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


def _moment_groups(static):
    """Groups of 32 voxel-major channels: the metric channels and the residual's real/imaginary parts."""
    P = static.basis_size
    metric = 1 if static.metric_trace_only else tri_size(P)
    return -(-(metric + 2 * P) // 32)


def _unpack_moments(moments, static):
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
    the layout :func:`relax.cuda.kernels.ppca_moment_scatter_f32` produces on GPU streams.
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
    metric_trace_only: bool = False,
    gemm_precision: str = "auto",
    device=None,
    tile_loader=None,
    pass2_mass_floor: float = 0.0,
) -> FullRowStream:
    """Upload the model, fine grids and priors once to ``device`` for an expectation's tiles.

    ``pass2_mass_floor`` (:attr:`FullRowStream.pass2_mass_floor`) lets pass 2 skip the pose rows
    that carry less posterior mass than it for every image of a tile.

    ``tile_loader`` reads the tiles of a dataset that is not a single-particle image dataset
    (:attr:`FullRowStream.tile_loader`); ``experiment_dataset`` then only supplies the image and
    volume shapes and the original ids of its images.

    With ``metric_trace_only`` the tiles return :class:`TracePPCAStats`: the
    LHS metric is backprojected only as its per-frequency trace.
    ``gemm_precision`` (:data:`GEMM_PRECISIONS`) sets the four stream GEMMs: ``"tf32"`` one
    TF32 tensor-core pass with float32 accumulation (GPUs of compute capability 8.0 or later),
    ``"fp32"`` exact float32 products, ``"auto"`` (the default) tf32 where available, else fp32
    (:func:`resolve_gemm_precision`); section 14 of ``docs/math/vdam_ppca_algorithm.md`` records
    the science comparison behind the default.
    """
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
            metric_trace_only=metric_trace_only,
            gemm_precision=gemm_precision,
            device=device,
            tile_loader=tile_loader,
        )._replace(pass2_mass_floor=float(pass2_mass_floor))


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
    metric_trace_only,
    gemm_precision,
    device,
    tile_loader,
) -> FullRowStream:
    gemm_precision = resolve_gemm_precision(gemm_precision, device)
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
    forward_config = (
        ForwardModelConfig.from_dataset(
            experiment_dataset, disc_type=disc_type, process_fn=experiment_dataset.process_images
        )
        if tile_loader is None
        else None
    )
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(noise_variance, resolved.image_shape).squeeze()
    block_size = int(schedule.rotation_block_size)
    block_starts = _block_starts(n_rot, block_size, device)
    # The sentinel row (identity rotation, zero prior) has the all-unsupported
    # coarse parent R_coarse: it pads the last block of every tile.
    sentinel_rotation = np.eye(3, dtype=np.float32)[None]
    cuda_kernels = device.platform == "gpu"
    arrays = _StreamArrays(
        augmented=resolved.augmented_half_volumes,
        augmented_voxel_major=jnp.asarray(resolved.augmented_half_volumes).T.copy() if cuda_kernels else None,
        gemm_window=_gemm_window(resolved.score_indices, resolved.image_shape) if cuda_kernels else None,
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
        metric_trace_only=bool(metric_trace_only),
        cuda_kernels=cuda_kernels,
        gemm_precision=str(gemm_precision),
    )
    if static.cuda_kernels and (decide_order(static.disc_type) != 1 or static.relion_texture_interp):
        raise ValueError("The CUDA stream kernels support linear interpolation without RELION texture lookup only")
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
        tile_loader=tile_loader,
    )


@functools.lru_cache(maxsize=8)
def _block_starts(n_rot: int, block_size: int, device) -> tuple[jax.Array, ...]:
    """Device scalars of every block start; built once per grid (hundreds of transfers at HP4)."""
    return tuple(jax.device_put(np.int32(start), device) for start in range(0, n_rot, block_size))


def _read_tile(stream: FullRowStream, image_indices, significant_rows, *, collect_observation: bool):
    """One tile's operands, observation power and layout, from the stream's tile reader."""
    if len(significant_rows) != np.asarray(image_indices).size:
        raise ValueError("One coarse support row is required per tile image")
    if np.asarray(image_indices).size > stream.image_batch_size:
        raise ValueError("A full-row tile must fit one image batch")
    loader = _load_tile if stream.tile_loader is None else stream.tile_loader
    return loader(stream, image_indices, significant_rows, collect_observation=collect_observation)


def tile_support(stream: FullRowStream, significant_rows):
    """Coarse support mask with its unsupported sentinel row, padded row table and layout of one tile."""
    coarse = coarse_support_mask(significant_rows, stream.n_coarse_rotations, stream.n_coarse_translations)
    rows = tile_fine_rows(coarse, stream.rotation_parent)
    n_blocks = -(-rows.size // stream.rotation_block_size)
    capacity = len(stream.block_starts) * stream.rotation_block_size
    table = np.full(capacity, stream.rotation_parent.size, dtype=np.int32)
    table[: rows.size] = rows
    unsupported = np.zeros((coarse.shape[0], 1, coarse.shape[2]), dtype=bool)
    # Exact per-image rows versus the scored union, for the wasted-work record.
    supported = np.any(coarse, axis=2)[:, stream.rotation_parent].sum()
    layout = {
        "rows": rows,
        "n_blocks": n_blocks,
        "scored_rows": int(n_blocks * stream.rotation_block_size),
        "supported_image_rows": int(supported),
    }
    return jnp.asarray(np.concatenate([coarse, unsupported], axis=1)), jnp.asarray(table), layout


def _load_tile(stream: FullRowStream, image_indices, significant_rows, *, collect_observation: bool):
    image_indices = np.asarray(image_indices)
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
    coarse_mask, table, layout = tile_support(stream, significant_rows)
    Y1, ctf2, Y1_recon, ctf2_recon = batch.Y1_score, batch.ctf2_score, batch.Y1_recon, batch.ctf2_recon
    shift_pad = 0
    if stream.static.cuda_kernels:
        # Zero operands at the padding pixels of the GEMM window and the padding (image,
        # translation) columns of the shift axis: they add exact zeros.
        pad = stream.arrays.gemm_window.shape[0] - ctf2.shape[1]
        Y1, Y1_recon = (jnp.pad(x, ((0, 0), (0, 0), (0, pad))) for x in (Y1, Y1_recon))
        ctf2, ctf2_recon = (jnp.pad(x, ((0, 0), (0, pad))) for x in (ctf2, ctf2_recon))
        shift_pad = -(Y1.shape[0] * Y1.shape[1]) % _SHIFT_ALIGN
    tile = _TileArrays(
        coarse_mask=coarse_mask,
        rows=table,
        Y1=jnp.pad(_real_imag(Y1).reshape(-1, 2 * ctf2.shape[1]).T, ((0, 0), (0, shift_pad))),
        ctf2=ctf2.T,
        Y1_recon=jnp.pad(_real_imag(Y1_recon).reshape(-1, 2 * ctf2_recon.shape[1]), ((0, shift_pad), (0, 0))),
        ctf2_recon=ctf2_recon,
        y_norm=batch.y_norm,
    )
    layout.update(
        n_observations=int(image_indices.size),
        original_ids=stream.dataset.original_image_indices_from_local(image_indices),
    )
    return tile, batch.observation_power, layout


def _score_tile(stream: FullRowStream, tile: _TileArrays, n_blocks: int, kept: _Kept | None = None):
    """Pass 1 and the tile normalization; returns the kept results and the posterior summary.

    ``kept`` may be the previous tile's buffer of the same shape: pass 1 overwrites
    every row of the first ``n_blocks`` blocks, and nothing reads the rows after them.
    """
    n_images = int(tile.y_norm.shape[0])
    T = int(stream.translations.shape[0])
    q = stream.static.basis_size - 1
    capacity = len(stream.block_starts) * stream.rotation_block_size
    if kept is None or kept.score.shape != (capacity, n_images, T):
        kept = _empty_kept(capacity, n_images, T, q, jnp.float32)
    for start in stream.block_starts[:n_blocks]:
        kept = _score_block(
            stream.arrays, tile, kept, start, static=stream.static, block_size=stream.rotation_block_size
        )
    return kept, _tile_posterior(kept, tile.rows, n_blocks, stream.rotation_block_size, stream.static)


# Device memory a tile plan leaves unplanned for allocator fragmentation and XLA program
# temporaries the plan does not count, as a share of the device's memory.
TILE_FRAGMENTATION_HEADROOM = 0.10


def _window_pixels(stream: FullRowStream) -> int:
    """Pixels per frame of the tile operands (the padded GEMM window on GPU streams)."""
    arrays = stream.arrays
    if arrays.gemm_window is not None:
        return int(arrays.gemm_window.shape[0])
    if arrays.score_indices is not None:
        return int(np.asarray(arrays.score_indices).size)
    return int(arrays.coefficient_noise.size)


# What a tile reader other than the single-particle one must report for the tile planner.
READER_PLAN_ATTRIBUTES = ("operand_bytes", "max_frames")


def _plan_reader(stream: FullRowStream):
    """The stream's custom tile reader, after checking it reports what the planner counts; None for
    the single-particle reader. A reader without them would be planned as single particles, which
    can run a card out of memory, so that is an error."""
    reader = stream.tile_loader
    if reader is None:
        return None
    missing = [name for name in READER_PLAN_ATTRIBUTES if not callable(getattr(reader, name, None))]
    if missing:
        raise ValueError(
            f"Tile reader {getattr(reader, '__qualname__', reader)!r} has no {', '.join(missing)}: the tile "
            "planner cannot count its memory (see READER_PLAN_ATTRIBUTES)"
        )
    return reader


def _tile_frames(stream: FullRowStream) -> int:
    """Projection frames per tile row: a custom reader's ``max_frames`` (a subtomogram reader's tilts), else 1."""
    reader = _plan_reader(stream)
    return 1 if reader is None else int(reader.max_frames(stream))


def _tile_spec(stream: FullRowStream, n_images: int) -> _TileArrays:
    """Shapes and dtypes of a tile of ``n_images`` as the stream's readers build it (:class:`_TileArrays`)."""
    B, T = int(n_images), int(stream.translations.shape[0])
    K, F = _tile_frames(stream), _window_pixels(stream)
    BT = B * T + (-(B * T) % _SHIFT_ALIGN if stream.static.cuda_kernels else 0)
    f32 = jnp.float32
    spec = jax.ShapeDtypeStruct
    return _TileArrays(
        coarse_mask=spec((B, stream.n_coarse_rotations + 1, stream.n_coarse_translations), jnp.bool_),
        rows=spec((len(stream.block_starts) * stream.rotation_block_size,), jnp.int32),
        Y1=spec((K * 2 * F, BT), f32),
        ctf2=spec((K * F, B), f32),
        Y1_recon=spec((BT, K * 2 * F), f32),
        ctf2_recon=spec((B, K * F), f32),
        y_norm=spec((B,), f32),
        frames=None if _plan_reader(stream) is None else spec((K, 3, 3), f32),
    )


_PROGRAM_BYTES: dict = {}


def tile_program_bytes(stream: FullRowStream, n_images: int) -> int:
    """Device bytes the pass-1 and pass-2 block programs add for a tile of ``n_images``, from XLA.

    The larger of :func:`_score_block` and :func:`_moment_block` (moments) at the tile's shapes:
    each program's temporaries plus the outputs that do not alias its donated input (the kept
    results, the moment carry), from ``lower(...).compile().memory_analysis()``. Cached per stage
    shape and size. These are separate compiles from the tiles' own (their operands' device
    placement differs, so the cache keys do): about 0.3 s per stage on A100 at EMPIAR-10076's shapes.
    """
    key = (_plan_shape_key(stream), int(n_images), getattr(stream.device, "id", None))
    if key not in _PROGRAM_BYTES:
        static, block = stream.static, stream.rotation_block_size
        capacity = len(stream.block_starts) * block
        T, q = int(stream.translations.shape[0]), static.basis_size - 1
        tile = _tile_spec(stream, n_images)
        kept = jax.eval_shape(lambda: _empty_kept(capacity, int(n_images), T, q, jnp.float32))
        carry = jax.eval_shape(lambda: _empty_carry(stream, int(n_images), jnp.float32(0)))
        vector = jax.ShapeDtypeStruct((int(n_images),), jnp.float32)
        index = jax.ShapeDtypeStruct((int(n_images),), jnp.int32)
        posterior = _Posterior(vector, vector, vector, index, index)
        start = jax.ShapeDtypeStruct((), jnp.int32)
        with jax.default_device(stream.device):
            programs = (
                _score_block.lower(stream.arrays, tile, kept, start, static=static, block_size=block),
                _moment_block.lower(
                    carry, stream.arrays, tile, kept, posterior, start, static=static, block_size=block, moments=True
                ),
            )
            analyses = [program.compile().memory_analysis() for program in programs]
        if any(analysis is None for analysis in analyses):
            raise RuntimeError("XLA reported no memory analysis for the stream's block programs")
        _PROGRAM_BYTES[key] = max(
            int(a.temp_size_in_bytes) + int(a.output_size_in_bytes) - int(a.alias_size_in_bytes) for a in analyses
        )
    return _PROGRAM_BYTES[key]


def stream_tile_bytes(stream: FullRowStream, n_images: int) -> int:
    """Device bytes of the stream's own buffers for a tile of ``n_images``.

    The kept pass-1 results over the full row table (:func:`_empty_kept`, from their shapes) with,
    under a pass-2 mass floor, its compacted copy of at most half the rows; the moment accumulators
    and their compensation (:func:`_empty_carry`); and the block programs' own memory
    (:func:`tile_program_bytes`: projections, GEMM outputs, weights and M-step images).
    """
    static = stream.static
    P = static.basis_size
    T = int(stream.translations.shape[0])
    capacity = len(stream.block_starts) * stream.rotation_block_size
    kept = capacity * (T * P + tri_size(P - 1) + 3)
    if stream.pass2_mass_floor > 0:
        kept = kept + kept // 2  # the compacted pass-2 copy of at most half the rows (_compact_capacity)
    accumulators = 2 * _moment_groups(static) * int(stream.arrays.augmented.shape[1]) * 32
    return 4 * (int(n_images) * kept + accumulators) + tile_program_bytes(stream, n_images)


def single_particle_reader_bytes(stream: FullRowStream, n_images: int) -> tuple[int, int]:
    """``(peak, resident)`` device bytes of :func:`_load_tile` for ``n_images``, from its arrays.

    Peak counts every array the reader creates as live at once: the shifted half images
    ``(B T, n_half)`` complex from ``preprocess_batch``, their scaled and masked score and
    reconstruction copies, the windowed copies, the GEMM-window padding, the planar ``[Re | Im]``
    stacks and their padded transposes, and the CTF rows. Resident is the tile's two planar operands
    and CTF rows, live until the tile finishes.
    """
    T = int(stream.translations.shape[0])
    n_half = int(stream.arrays.coefficient_noise.size)
    F = _window_pixels(stream)
    complex_full = 2 * T * n_half  # one (T, n_half) complex64 image stack, in float32 values
    complex_window = 2 * T * F
    planar = 2 * T * F
    peak = 3 * complex_full + 4 * complex_window + 4 * planar + 6 * n_half
    resident = 2 * planar + 2 * F
    return 4 * int(n_images) * peak, 4 * int(n_images) * resident


def _reader_bytes(stream: FullRowStream):
    """The tile reader's ``(peak, resident)`` byte count: a custom reader's ``operand_bytes`` (the
    subtomogram reader's, from XLA's compiled memory analysis of its operand program), else
    :func:`single_particle_reader_bytes`."""
    reader = _plan_reader(stream)
    return single_particle_reader_bytes if reader is None else reader.operand_bytes


def tile_bytes(stream: FullRowStream, n_images: int, reader=None, *, pipelined: bool = True) -> int:
    """Device bytes a tile of ``n_images`` needs at its peak.

    Tiles of one :func:`accumulate_full_row_tiles` call are pipelined: the next tile's reader runs
    while the current tile's operands and the shared kept buffer are live, so the count is the
    stream's own buffers (:func:`stream_tile_bytes`), one tile's resident operands and one reader
    peak. A call of one tile (``pipelined=False``) reads it before its passes run: the larger of the
    reader peak and the stream's buffers with the tile's operands.
    """
    peak, resident = (reader or _reader_bytes(stream))(stream, n_images)
    own = stream_tile_bytes(stream, n_images) + resident
    return own + peak if pipelined else max(own, peak)


def plan_tile_images(
    stream: FullRowStream, requested: int, *, memory_bytes=None, device_bytes=None, tiles_per_call=None
) -> int:
    """Tile images (particles) per tile: ``requested``, or the most that fit the device (:func:`tile_bytes`).

    The budget is ``memory_bytes`` minus :data:`TILE_FRAGMENTATION_HEADROOM` of ``device_bytes``. On a
    GPU stream they default to what the device can still hand out after the stream's upload
    (:func:`relax.sparse_pass2.sparse_pass2_budget.device_available_bytes`) and the device's memory;
    on a CPU stream to the host memory left to this process and the host memory it may use
    (:mod:`relax.helpers.host_memory`). Every stage (radius, window, pose grid) is planned with its
    own shapes, once: with the probed defaults the plan is kept per stage shape, measured at the
    stage's first update. A stage whose one-image tile does not fit is refused: its rotation block
    is too large for the device.

    ``tiles_per_call(size)`` gives the most tiles one :func:`accumulate_full_row_tiles` call holds at
    a tile size (never fewer for smaller tiles). Tiles are read ahead only within a call, so when
    every call holds one tile at the plan without read-ahead, that plan is used; otherwise (or when
    it is None) the next tile's reader is counted.
    """
    if tiles_per_call is not None and tiles_per_call(int(requested)) <= 1:
        single = _planned_tile_images(stream, requested, memory_bytes, device_bytes, pipelined=False)
        if tiles_per_call(single) <= 1:
            return single
    return _planned_tile_images(stream, requested, memory_bytes, device_bytes, pipelined=True)


def _planned_tile_images(stream, requested, memory_bytes, device_bytes, *, pipelined):
    if memory_bytes is None and device_bytes is None:
        # The device probes run nvidia-smi (tens of ms); a stage's shapes are planned once.
        key = (_plan_shape_key(stream), int(requested), stream.device.id, pipelined)
        if key not in _PLANS:
            _PLANS[key] = _planned_tile_images(stream, requested, *_available_bytes(stream), pipelined=pipelined)
            if stream.static.cuda_kernels:
                reserve_plan_region(stream.device, tile_bytes(stream, _PLANS[key], pipelined=pipelined))
        return _PLANS[key]
    if not memory_bytes:
        return int(requested)
    if device_bytes is None and stream.static.cuda_kernels:
        from relax.sparse_pass2 import sparse_pass2_budget as budget

        device_bytes = budget._device_memory_limit_bytes()
    budget_bytes = memory_bytes - TILE_FRAGMENTATION_HEADROOM * (device_bytes or memory_bytes)
    reader = _reader_bytes(stream)
    requested = int(requested)
    planned = _plan_tile_images(stream, requested, reader, budget_bytes, pipelined)
    peak, resident = reader(stream, planned)
    gib = 2**30
    counted = tile_bytes(stream, planned, reader, pipelined=pipelined)
    if counted > budget_bytes:
        raise ValueError(
            f"A tile of one image needs {counted / gib:.2f} GiB on the {stream.device.platform} "
            f"(block programs {tile_program_bytes(stream, planned) / gib:.2f} GiB at rotation block "
            f"{stream.rotation_block_size}), more than its {budget_bytes / gib:.2f} GiB budget: "
            "use a smaller rotation block"
        )
    logger.info(
        "PPCA tile plan: %d of %d images per tile (%d frames each); counted %.2f GiB of a %.2f GiB budget "
        "(%.2f GiB available, %.2f GiB device, headroom %.0f%%): stream %.2f GiB of which block programs "
        "%.2f GiB, tile operands %.2f GiB, reader %.2f GiB (%s)",
        planned,
        requested,
        _tile_frames(stream),
        counted / gib,
        budget_bytes / gib,
        memory_bytes / gib,
        (device_bytes or memory_bytes) / gib,
        100 * TILE_FRAGMENTATION_HEADROOM,
        stream_tile_bytes(stream, planned) / gib,
        tile_program_bytes(stream, planned) / gib,
        resident / gib,
        peak / gib,
        "next tile read ahead" if pipelined else "one tile per call",
    )
    return planned


# Plans by stage shapes (:func:`_plan_shape_key`) for the device probes' defaults.
_PLANS: dict = {}


def _plan_shape_key(stream: FullRowStream) -> tuple:
    """Everything :func:`tile_bytes` reads from a stream: a stage's pose grid, window, rank and frames."""
    static = stream.static
    return (
        len(stream.block_starts),
        stream.rotation_block_size,
        int(stream.translations.shape[0]),
        _window_pixels(stream),
        int(stream.arrays.coefficient_noise.size),
        int(stream.arrays.augmented.shape[1]),
        static.basis_size,
        static.metric_trace_only,
        _tile_frames(stream),
        stream.tile_loader,
    )


def _available_bytes(stream):
    """``(available, total)`` memory for the stream's tiles: the GPU's after the stream's upload and
    its memory, or the host memory left to this process and the host memory it may use."""
    if not stream.static.cuda_kernels:
        from relax.helpers.host_memory import available_memory_bytes, resident_bytes

        host = available_memory_bytes()
        return host - resident_bytes(), host
    from relax.sparse_pass2 import sparse_pass2_budget as budget

    available = budget.device_available_bytes(
        budget._device_free_memory_bytes(),
        budget._jax_allocator_free_memory_bytes(),
        budget._jax_allocator_pool_free_bytes(),
    )
    return available, None


# Memory a plan's region leaves to the device (allocator rounding and the device's other users).
PLAN_REGION_MARGIN_BYTES = 256 * 2**20


def plan_region_bytes(counted_bytes, limit_bytes, pool_bytes, physical_free_bytes) -> int:
    """Size of the one block that gives a plan's counted bytes contiguous memory in the allocator's pool.

    The counted bytes, or, when the allocator may no longer take that much from the device, all it
    still may: its limit minus its pool, bounded by the device's free memory, less
    :data:`PLAN_REGION_MARGIN_BYTES`. Zero when a reading is missing or nothing is left to take (a
    preallocated pool, or a pool earlier plans grew to its limit).
    """
    if limit_bytes is None or pool_bytes is None or physical_free_bytes is None:
        return 0
    unreserved = min(int(limit_bytes) - int(pool_bytes), int(physical_free_bytes)) - PLAN_REGION_MARGIN_BYTES
    return max(0, min(int(counted_bytes), unreserved))


def reserve_plan_region(device, counted_bytes) -> int:
    """Give a plan's counted bytes one contiguous region of the GPU allocator's pool; returns the block's bytes.

    A plan counts bytes, and the allocator hands out a block only from one contiguous region.
    Without preallocation (``XLA_PYTHON_CLIENT_PREALLOCATE=false``) the pool grows by regions sized
    by the requests so far, so the small early stages leave a pool of small regions in which a later
    stage's large block finds no room while its bytes are free: an 8.48 GiB tilt-reader block failed
    with 9.2 GiB in use of a 36.4 GiB pool on a 40 GB A100 (Polar 413542; reproduced on an H100
    limited to that pool, Slurm 14992893, where the same run with a preallocated pool passes).
    Allocating :func:`plan_region_bytes` as one block and releasing it makes the allocator grow the
    pool by a region of at least that size, unless the pool already has such a block free. The plan
    and the results do not change: the allocator can hand out the same bytes before and after, and
    the pool only takes earlier what the planned tiles would take.
    """
    from relax.sparse_pass2 import sparse_pass2_budget as budget

    stats = device.memory_stats() or {}
    size = plan_region_bytes(
        counted_bytes, stats.get("bytes_limit"), stats.get("pool_bytes"), budget._device_free_memory_bytes()
    )
    megabytes = size // 2**20
    if megabytes == 0:
        return 0
    with jax.default_device(device):
        jnp.zeros((megabytes, 2**20), jnp.uint8).block_until_ready()
    grown = int(device.memory_stats()["pool_bytes"]) - int(stats["pool_bytes"])
    if grown:
        logger.info(
            "PPCA tile plan: the allocator pool grew by %.2f GiB for one block of %.2f GiB",
            grown / 2**30,
            megabytes / 2**10,
        )
    return megabytes * 2**20


def _plan_tile_images(stream, requested, reader, budget_bytes, pipelined):
    exact = {requested: tile_bytes(stream, requested, reader, pipelined=pipelined)}
    if exact[requested] <= budget_bytes:
        return requested
    # Tile bytes are affine in the tile size (every counted buffer and program grows linearly in the
    # images): two exact counts (1 and ``requested``) place every size between them, so the search
    # compiles no other program. The chosen size is then counted exactly.
    exact[1] = tile_bytes(stream, 1, reader, pipelined=pipelined)
    slope = (exact[requested] - exact[1]) / max(requested - 1, 1)
    low = max(1, min(requested, 1 + int((budget_bytes - exact[1]) // slope) if slope > 0 else requested))
    while low > 1:
        exact.setdefault(low, tile_bytes(stream, low, reader, pipelined=pipelined))
        if exact[low] <= budget_bytes:
            break
        low = max(1, min(low - 1, int(low * budget_bytes / exact[low])))
    return low


def _empty_kept(capacity: int, n_images: int, n_translations: int, q: int, dtype) -> _Kept:
    return _Kept(
        score=jnp.full((capacity, n_images, n_translations), -jnp.inf, dtype),
        latent_mean=jnp.zeros((q, capacity, n_images, n_translations), dtype),
        latent_covariance=jnp.zeros((capacity, n_images, tri_size(q)), dtype),
        part_max=jnp.full((capacity, n_images), -jnp.inf, dtype),
        part_arg=jnp.zeros((capacity, n_images), jnp.int32),
        part_sum=jnp.zeros((capacity, n_images), dtype),
    )


def _tile_posterior(kept: _Kept, rows, n_blocks: int, block_size: int, static: _StreamStatic) -> _Posterior:
    if static.cuda_kernels:
        return _normalize_partials(kept, rows, n_blocks=n_blocks, block_size=block_size)
    return _normalize(kept.score, rows, n_blocks=n_blocks, block_size=block_size)


def _check_finite_posterior(posterior: _Posterior, n_real: int | None = None):
    center, centered_logZ = posterior.center[:n_real], posterior.centered_logZ[:n_real]
    finite = jax.device_get((jnp.all(jnp.isfinite(center)), jnp.all(jnp.isfinite(centered_logZ))))
    if not all(finite):
        raise ValueError("Every full-row image needs a finite supported pose and partition")


def _run_pass2(stream, tile, kept, posterior, n_blocks, carry, *, moments):
    for start in stream.block_starts[:n_blocks]:
        carry = _moment_block(
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


def _empty_carry(stream, n_images, observation_power):
    arrays, static = stream.arrays, stream.static
    P = static.basis_size
    half_size = int(arrays.augmented.shape[1])
    capacity = len(stream.block_starts) * stream.rotation_block_size
    moments = (_moment_groups(static), half_size, 32)
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
    with ``collect_residuals=True``, except the augmented RHS volume
    (``rhs`` is ``None``): both InitialModel optimizers read only the LHS
    metric and the direct residual gradient. A ``metric_trace_only`` stream
    returns :class:`TracePPCAStats`, whose metric is the LHS trace alone.
    """
    return accumulate_full_row_tiles(stream, [(image_indices, significant_rows)], enforce_x0=enforce_x0)[0]


@full_float32
def accumulate_full_row_tiles(stream: FullRowStream, tiles, *, enforce_x0: bool = True) -> list[AugmentedPPCAStats]:
    """:func:`accumulate_full_row_tile` for each ``(image_indices, significant_rows)`` in ``tiles``.

    The tiles run largest first (results come back in the given order): the largest pose-kept buffer
    is allocated once, while the device memory is unfragmented, and reused by every tile of its size;
    a tile of another size releases it before allocating its own, so a single tile's worth is live
    (what :func:`stream_tile_bytes` counts). In the given order, a batch of 101 + 49 + 101 + 49 images
    reallocated the large buffer between smaller ones, and its compacted pass-2 copy (16 GiB in one
    array at the eleven-state HEALPix 4 stage) no longer found a contiguous block on an 80 GB card.
    While the device runs a tile's passes, the host finishes the tile before it (its statistics are
    complete) and reads and preprocesses the next one.
    """
    order = sorted(range(len(tiles)), key=lambda k: -np.asarray(tiles[k][0]).size)  # stable: ties keep their order
    with jax.default_device(stream.device):
        loaded = _read_tile(stream, *tiles[order[0]], collect_observation=True) if tiles else None
        results, kept, previous = [None] * len(tiles), None, None
        for position, index in enumerate(order):
            if kept is not None and kept.score.shape[1] != loaded[0].y_norm.shape[0]:
                # A tile of another size gets its own kept buffer: release this one first, so one is live.
                kept = None
            pending, kept = _enqueue_full_row_tile(stream, *loaded, kept)
            loaded = None
            if previous is not None:
                results[previous[0]] = _finish_full_row_tile(stream, *previous[1:], enforce_x0=enforce_x0)
            # Only this tile's operands stay live while the next one is read (tile_bytes counts that).
            previous = None
            if position + 1 < len(order):
                loaded = _read_tile(stream, *tiles[order[position + 1]], collect_observation=True)
            previous = (index, tiles[index][0], *pending)
        if previous is not None:
            results[previous[0]] = _finish_full_row_tile(stream, *previous[1:], enforce_x0=enforce_x0)
        return results


# Compiled tile sizes per stage: a padding reader rounds a tile up to one of these many sizes.
TILE_SIZE_BUCKETS = 5


def tile_size_bucket(n_images: int, planned: int) -> int:
    """The padded size of a tile of ``n_images``: the smallest of ``planned / 2^k`` (k < :data:`TILE_SIZE_BUCKETS`,
    rounded up) that holds it, so a stage compiles at most :data:`TILE_SIZE_BUCKETS` tile shapes."""
    if not 0 < n_images <= planned:
        raise ValueError("A tile holds between 1 and the planned number of images")
    sizes = sorted({max(1, -(-int(planned) // (1 << k))) for k in range(TILE_SIZE_BUCKETS)})
    return next(size for size in sizes if size >= n_images)


def _n_real(tile, layout) -> int:
    """Real images of a tile: a reader may pad it with dummy images to a fixed size (``layout["n_real"]``)."""
    return int(layout.get("n_real", tile.y_norm.shape[0]))


def _drop_padding(posterior: _Posterior, n_real: int) -> _Posterior:
    """Give a tile's padding images no posterior mass: an infinite normalizer makes every pass-2 weight
    (embeddings, moment sums, rotation mass, entropy, offsets, counts, row masses) of those images 0.

    A reader pads a tile with dummy images (every operand zero) so the tile keeps a compiled size; real
    images come first.
    """
    if n_real == posterior.center.shape[0]:
        return posterior
    real = jnp.arange(posterior.center.shape[0]) < n_real
    return posterior._replace(center=jnp.where(real, posterior.center, jnp.inf))


def _enqueue_full_row_tile(stream, tile, observation_power, layout, kept=None):
    """Dispatch both passes of one loaded tile; returns its pass-1 kept buffer too.

    The device runs ahead of the host, except for one wait per tile when pass 2 skips rows
    (:func:`_pass2_rows` needs the row count).
    """
    kept, posterior = _score_tile(stream, tile, layout["n_blocks"], kept)
    posterior = _drop_padding(posterior, _n_real(tile, layout))
    carry = _empty_carry(stream, int(tile.y_norm.shape[0]), observation_power)
    carry = _run_skipping_pass2(stream, tile, kept, posterior, layout, carry, moments=True)
    return (tile, layout, posterior, carry), kept


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


def _run_skipping_pass2(stream, tile, kept, posterior, layout, carry, *, moments):
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


def _finish_full_row_tile(stream, image_indices, tile, layout, posterior, carry, *, enforce_x0):
    n_images = _n_real(tile, layout)
    _check_finite_posterior(posterior, n_images)
    static = stream.static
    lhs_tri, residual = _unpack_moments(carry.moments, static)
    if enforce_x0:
        lhs_tri = _enforce_augmented_x0(lhs_tri.astype(jnp.complex64), static.volume_shape).real.astype(jnp.float32)
    weights = make_half_image_weights(static.image_shape)
    shells = make_shell_indices_half(static.image_shape)
    n_shells = int(np.max(np.asarray(shells))) + 1
    residual_num = jnp.zeros(n_shells, jnp.float32).at[shells].add(weights * carry.residual_power)
    # Every observed image counts once: a subtomogram particle observes each of its visible tilts.
    residual_den = jnp.zeros(n_shells, jnp.float32).at[shells].add(weights * layout["n_observations"])
    # Scores exclude the pose-invariant image constant; the absolute log-partition adds it back.
    logZ = posterior.center + posterior.centered_logZ + pose_invariant_score_offset(tile.y_norm)
    real = slice(0, n_images)  # padding images (after the real ones) report nothing
    host = jax.device_get(
        {
            "log_likelihood": jnp.sum(logZ[real]),
            "top_centered_score": (posterior.top_score - posterior.center)[real],
            "centered_logZ": posterior.centered_logZ[real],
            "rotation": posterior.top_rotation[real],
            "translation": posterior.top_translation[real],
            "n_significant": carry.n_significant[real],
            "rotation_mass": carry.rotation_mass,
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
    original_ids = layout["original_ids"]
    diagnostics = {
        "engine": FULL_ROW_ENGINE,
        "gemm_precision": static.gemm_precision,
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
        "tile_size": int(tile.y_norm.shape[0]),  # with padding: the compiled tile shape
        "scored_image_rows": n_images * layout["scored_rows"],
        "supported_image_rows": layout["supported_image_rows"],
        # Pose rows pass 2 visited (all scored rows unless pass2_mass_floor skips some).
        "pass2_rows": layout["pass2_rows"],
        "scored_rows": layout["scored_rows"],
        "latent_covariance_trace_mean": float(host["latent"] / np.float32(n_images)),
        "pose_entropy_mean": float(host["entropy"] / np.float32(n_images)),
    }
    fields = dict(
        # No RHS volume: the streamed statistics feed the direct residual gradient only.
        rhs=None,
        log_likelihood=float(host["log_likelihood"]),
        n_images=n_images,
        residual_gradient=residual,
        residual_num=residual_num,
        residual_den=residual_den,
        embeddings=carry.embedding[real],
        original_image_ids=original_ids,
        diagnostics=diagnostics,
    )
    if static.metric_trace_only:
        return TracePPCAStats(lhs_tri=None, metric_trace=lhs_tri[0], **fields)
    return AugmentedPPCAStats(lhs_tri=jnp.swapaxes(lhs_tri, 0, 1), **fields)


@full_float32
def full_row_tile_embeddings(stream: FullRowStream, image_indices, significant_rows) -> DensePPCAEmbeddings:
    """Pose-marginal embeddings of one full-row tile, as ``compute_dense_ppca_embeddings``."""
    with jax.default_device(stream.device):
        tile, _, layout = _read_tile(stream, image_indices, significant_rows, collect_observation=False)
        kept, posterior = _score_tile(stream, tile, layout["n_blocks"])
        n_real = _n_real(tile, layout)
        _check_finite_posterior(posterior, n_real)
        posterior = _drop_padding(posterior, n_real)
        carry = _empty_carry(stream, int(tile.y_norm.shape[0]), jnp.float32(0))
        carry = _run_skipping_pass2(stream, tile, kept, posterior, layout, carry, moments=False)
        return DensePPCAEmbeddings(carry.embedding[:n_real], layout["original_ids"], n_real)
