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

1. Pass 1, per rotation block (:func:`full_row_kernels.score_block`): one real GEMM of the
   augmented projections against the shifted images gives every pose's
   inner products, one of the projection products against the CTF/noise
   weights gives the latent Gram; the pose scores, latent means and latent
   covariances (``full_row_kernels._latent_block``) are kept on the device for the tile.
2. The tile's maxima, centered log-partitions and top poses follow from the
   kept scores (``full_row_kernels._normalize``).
3. Pass 2, per rotation block (:func:`full_row_kernels.moment_block`): the posterior weights
   and the kept moments form the M-step images with two more GEMMs (weighted
   images and weighted CTF/noise), then expected residuals, diagnostics and
   the augmented backprojection. Nothing of pass 1 is recomputed except the
   projections the residuals need.

On GPU streams the elementwise stages around the GEMMs run in relax's CUDA
kernels (``full_row_kernels._score_block_cuda``, ``_posterior_block_cuda``, the window
projector and the moment scatter); the XLA formulation is the CPU path and the
tests' float64 reference.

Kept operands are component-major ``(component, R, B, T)``, the layout of the
projector's output and of the backprojector's input, so each GEMM reads and
writes them in place and no operand is transposed. Every posterior weight is
used: float32 weights of these posteriors are nonzero for essentially every
(image, rotation) pair.

The kernels and records are in ``full_row_kernels``, the tile memory planner in ``full_row_tile_plan``;
this module prepares the stream, reads tiles and drives the two passes. The formulation is section 14 of
``docs/math/vdam_ppca_algorithm.md``.
Tests compare the engine against the host-mask reference
:func:`relax.ppca_refinement.dense_dataset.accumulate_dense_ppca_statistics`.
"""

from __future__ import annotations

import dataclasses
import functools
import logging

import jax
import jax.numpy as jnp
import numpy as np
from recovar.core.configs import ForwardModelConfig
from recovar.core.slicing import decide_order
from recovar.ppca.pose_accumulators import AugmentedPPCAStats
from recovar.reconstruction import noise as noise_utils

from relax.fourier.half_spectrum import make_half_image_weights, make_shell_indices_half
from relax.ppca_refinement.config import GeometryConfig, ScheduleConfig, ScoringConfig
from relax.ppca_refinement.dense_blocks import prepare_dense_ppca_dataset_inputs, prepare_dense_ppca_image_batch
from relax.ppca_refinement.dense_dataset import DensePPCAEmbeddings
from relax.ppca_refinement.engine import (
    enforce_augmented_x0,
    pose_invariant_score_offset,
)
from relax.ppca_refinement.full_row_kernels import (
    SHIFT_ALIGN,
    FullRowStream,
    Posterior,
    StreamArrays,
    StreamStatic,
    TileArrays,
    empty_carry,
    real_imag,
    resolve_gemm_precision,
    run_skipping_pass2,
    score_tile,
    unpack_moments,
)
from relax.ppca_refinement.residual_statistics import full_float32

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


# Window pixels per GEMM row multiple on GPU streams: unaligned fp32 operands (2F = 3002 at
# 10076) select cuBLAS's align1 kernels, about 4% slower on A100 (jobs/local_*/gemm_align).
_GEMM_ALIGN = 4


def coarse_support_mask(significant_rows, n_coarse_rotations: int, n_coarse_translations: int) -> np.ndarray:
    """Return ``(B, R_coarse, T_coarse)`` coarse support from packed significant ids.

    ``None`` keeps every coarse pose, as in
    :func:`relax.local_search.layout.build_pass2_hypothesis_layout`.
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


def _gemm_window(score_indices, image_shape) -> jax.Array:
    """The score window padded with -1 (no pixel) to a multiple of :data:`_GEMM_ALIGN`."""
    n_half = int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)
    window = np.arange(n_half, dtype=np.int32) if score_indices is None else np.asarray(score_indices, np.int32)
    pad = -window.size % _GEMM_ALIGN
    return jnp.asarray(np.concatenate([window, np.full(pad, -1, np.int32)]))


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
    block_starts = device_block_starts(n_rot, block_size, device)
    # The sentinel row (identity rotation, zero prior) has the all-unsupported
    # coarse parent R_coarse: it pads the last block of every tile.
    sentinel_rotation = np.eye(3, dtype=np.float32)[None]
    cuda_kernels = device.platform == "gpu"
    arrays = StreamArrays(
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
    static = StreamStatic(
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
def device_block_starts(n_rot: int, block_size: int, device) -> tuple[jax.Array, ...]:
    """Device scalars of every block start; built once per grid (hundreds of transfers at HP4)."""
    return tuple(jax.device_put(np.int32(start), device) for start in range(0, n_rot, block_size))


def read_tile(stream: FullRowStream, image_indices, significant_rows, *, collect_observation: bool):
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
    # With a planned tile size on the stream, the tile is padded to its :func:`tile_size_bucket` with
    # zero images after the real ones (``n_real``), as the subtomogram reader pads, so a stage compiles
    # one program per bucket instead of one per remainder size. Padding images carry the first
    # image's CTF parameters through preprocessing; their operands are zeroed below.
    n_real = int(image_indices.size)
    padded = n_real if stream.tile_images is None else tile_size_bucket(n_real, stream.tile_images)
    if padded != n_real:
        extra = padded - n_real
        batch_data = np.concatenate([batch_data, np.zeros((extra, *batch_data.shape[1:]), batch_data.dtype)])
        ctf_params = np.concatenate([ctf_params, np.repeat(np.asarray(ctf_params)[:1], extra, axis=0)])
        indices = np.concatenate([indices, np.repeat(np.asarray(indices)[:1], extra)])
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
    if padded != n_real:
        real = jnp.arange(padded) < n_real
        ctf2, ctf2_recon = (jnp.where(real[:, None], x, 0) for x in (ctf2, ctf2_recon))
        # Padding images take the first image's support; the stream gives them no posterior mass.
        coarse_mask = jnp.concatenate([coarse_mask, jnp.repeat(coarse_mask[:1], padded - n_real, axis=0)])
    shift_pad = 0
    if stream.static.cuda_kernels:
        # Zero operands at the padding pixels of the GEMM window and the padding (image,
        # translation) columns of the shift axis: they add exact zeros.
        pad = stream.arrays.gemm_window.shape[0] - ctf2.shape[1]
        Y1, Y1_recon = (jnp.pad(x, ((0, 0), (0, 0), (0, pad))) for x in (Y1, Y1_recon))
        ctf2, ctf2_recon = (jnp.pad(x, ((0, 0), (0, pad))) for x in (ctf2, ctf2_recon))
        shift_pad = -(Y1.shape[0] * Y1.shape[1]) % SHIFT_ALIGN
    tile = TileArrays(
        coarse_mask=coarse_mask,
        rows=table,
        Y1=jnp.pad(real_imag(Y1).reshape(-1, 2 * ctf2.shape[1]).T, ((0, 0), (0, shift_pad))),
        ctf2=ctf2.T,
        Y1_recon=jnp.pad(real_imag(Y1_recon).reshape(-1, 2 * ctf2_recon.shape[1]), ((0, shift_pad), (0, 0))),
        ctf2_recon=ctf2_recon,
        y_norm=batch.y_norm,
    )
    layout.update(
        n_real=n_real,
        n_observations=n_real,
        original_ids=stream.dataset.original_image_indices_from_local(image_indices),
    )
    return tile, batch.observation_power, layout


def check_finite_posterior(posterior: Posterior, n_real: int | None = None):
    center, centered_logZ = posterior.center[:n_real], posterior.centered_logZ[:n_real]
    finite = jax.device_get((jnp.all(jnp.isfinite(center)), jnp.all(jnp.isfinite(centered_logZ))))
    if not all(finite):
        raise ValueError("Every full-row image needs a finite supported pose and partition")


def _top_pose_posterior(top_centered_score: np.ndarray, centered_logZ: np.ndarray) -> np.ndarray:
    """Host float64 top-1 posterior, as in ``select_distinct_top_poses``."""
    with np.errstate(over="ignore", invalid="ignore"):
        posterior = np.exp(top_centered_score.astype(np.float64) - centered_logZ.astype(np.float64)[:, None])
    return np.where(np.isfinite(top_centered_score), posterior, 0.0).astype(np.float32)


@full_float32
def accumulate_full_row_tiles(stream: FullRowStream, tiles, *, enforce_x0: bool = True) -> list[AugmentedPPCAStats]:
    """Accumulate unregularized PPCA statistics for each ``(image_indices, significant_rows)`` in ``tiles``.

    ``significant_rows`` holds the packed coarse significant pose ids (or ``None``) of each tile image.
    Each result has the same statistics and summary diagnostics as the host-mask
    ``accumulate_dense_ppca_statistics`` call with ``collect_residuals=True``, except the augmented RHS
    volume (``rhs`` is ``None``): both InitialModel optimizers read only the LHS metric and the direct
    residual gradient. A ``metric_trace_only`` stream returns :class:`TracePPCAStats`, whose metric is the
    LHS trace alone.

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
        loaded = read_tile(stream, *tiles[order[0]], collect_observation=True) if tiles else None
        results, kept, previous = [None] * len(tiles), None, None
        for position, index in enumerate(order):
            if kept is not None and kept.score.shape[1] != loaded[0].y_norm.shape[0]:
                # A tile of another size gets its own kept buffer: release this one first, so one is live.
                kept = None
            pending, kept = _enqueue_full_row_tile(stream, *loaded, kept)
            loaded = None
            if previous is not None:
                results[previous[0]] = finish_full_row_tile(stream, *previous[1:], enforce_x0=enforce_x0)
            # Only this tile's operands stay live while the next one is read (tile_bytes counts that).
            previous = None
            if position + 1 < len(order):
                loaded = read_tile(stream, *tiles[order[position + 1]], collect_observation=True)
            previous = (index, tiles[index][0], *pending)
        if previous is not None:
            results[previous[0]] = finish_full_row_tile(stream, *previous[1:], enforce_x0=enforce_x0)
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


def real_image_count(tile, layout) -> int:
    """Real images of a tile: a reader may pad it with dummy images to a fixed size (``layout["n_real"]``)."""
    return int(layout.get("n_real", tile.y_norm.shape[0]))


def drop_padding(posterior: Posterior, n_real: int) -> Posterior:
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
    kept, posterior = score_tile(stream, tile, layout["n_blocks"], kept)
    posterior = drop_padding(posterior, real_image_count(tile, layout))
    carry = empty_carry(stream, int(tile.y_norm.shape[0]), observation_power)
    carry = run_skipping_pass2(stream, tile, kept, posterior, layout, carry, moments=True)
    return (tile, layout, posterior, carry), kept


def finish_full_row_tile(stream, image_indices, tile, layout, posterior, carry, *, enforce_x0):
    n_images = real_image_count(tile, layout)
    check_finite_posterior(posterior, n_images)
    static = stream.static
    lhs_tri, residual = unpack_moments(carry.moments, static)
    if enforce_x0:
        lhs_tri = enforce_augmented_x0(lhs_tri.astype(jnp.complex64), static.volume_shape).real.astype(jnp.float32)
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
        tile, _, layout = read_tile(stream, image_indices, significant_rows, collect_observation=False)
        kept, posterior = score_tile(stream, tile, layout["n_blocks"])
        n_real = real_image_count(tile, layout)
        check_finite_posterior(posterior, n_real)
        posterior = drop_padding(posterior, n_real)
        carry = empty_carry(stream, int(tile.y_norm.shape[0]), jnp.float32(0))
        carry = run_skipping_pass2(stream, tile, kept, posterior, layout, carry, moments=False)
        return DensePPCAEmbeddings(carry.embedding[:n_real], layout["original_ids"], n_real)
