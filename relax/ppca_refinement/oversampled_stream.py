"""PPCA adaptive oversampling: RELION's two passes on the streamed engine.

Section 14 of ``docs/math/vdam_ppca_algorithm.md`` ("Adaptive oversampling") states the model. Pass 1 scores every
coarse pose of a tile, the dense stream at the stage's HEALPix order
(:mod:`relax.ppca_refinement.full_row_stream`). Each image then keeps its significant coarse (rotation, translation)
samples: the largest posterior weights until their sum exceeds ``adaptive_fraction`` of the image's mass, at most
``max_significant`` of them, and every one of those weights at least the last one kept (RELION's significance pass,
``ml_optimiser.cpp:9602-9716``). Pass 2 scores, per image, only the children of its significant samples: the child
rotations (HEALPix N+1 directions and half psi steps, 8 per coarse rotation) times the child translations (half
steps, 4 per coarse translation in 2D and 8 in 3D), each with its coarse parent's priors (RELION evaluates the
orientation and offset priors at the coarse sample, ``ml_optimiser.cpp:9434``). The posterior over those poses and
the latent variable, and every statistic, are the dense stream's restricted to them: the result equals the dense
stream over the child grid with each image's significant coarse samples as its coarse support
(``tests/unit/ppca_initial_model/test_oversampled_stream.py``).

Pass 1 may run on a smaller image window than pass 2 (RELION's ``image_coarse_size``, matched to the coarse angular
step; :func:`relax.fourier.resolution.compute_coarse_image_size`): ``pass1`` and ``pass2`` are then two streams of the
same model and grids at the two window sizes. A tile's images are read once, unshifted at the pass-2 window; the
pass-1 operands are those pixels of them inside the pass-1 window times the coarse translations' phase factors
(:func:`_pass1_tile`).

Each significant sample is one job: an image, a coarse rotation and a coarse translation, scored at its children.
A tile's samples have image-major slots, ``max_significant`` per image (most unused); only the used ones are
scored, in fixed-size chunks that write their results at their slots, so every program keeps one shape per tile
size. A job's child-shifted images are formed in the job's program from the tile's
unshifted operands and the reader's phase factors (``shift_phases``), so a tile never holds its images at every
child translation.

Pass 2 accumulates, per image, only its significant fine samples: the same rule over the child weights
(``fine_fraction``, the adaptive fraction unless given; RELION's ``exp_significant_weight`` in
``storeWeightedSums``), with the image's normalization over every scored child. A child rotation of a job none of
whose translations is significant for its image is not visited: the M-step images are formed and backprojected per
kept (image, child rotation) row, in fixed-size chunks. With ``fine_fraction`` 1 every scored child is accumulated.
"""

from __future__ import annotations

import logging
from functools import lru_cache, partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from recovar.ppca.triangular import tri_size

from relax import sampling
from relax.cuda.kernels import ppca_window_project_f32
from relax.fourier.preprocessing import half_translation_phase_table
from relax.ppca_refinement.dense_dataset import DensePPCAEmbeddings
from relax.ppca_refinement.full_row_stream import (
    _HIGHEST,
    _SHIFT_ALIGN,
    TILE_FRAGMENTATION_HEADROOM,
    FullRowStream,
    _available_bytes,
    _block_starts,
    _check_finite_posterior,
    _drop_padding,
    _empty_carry,
    _finish_full_row_tile,
    _frame_rotations,
    _gemm,
    _latent_scores,
    _n_frames,
    _n_real,
    _plan_shape_key,
    _Posterior,
    _project,
    _read_tile,
    _real_imag,
    _scatter_moment_images,
    _score_tile,
    _second_moment_sums,
    _tile_frames,
    _tile_spec,
    _window,
    tile_support,
)
from relax.ppca_refinement.residual_statistics import full_float32

logger = logging.getLogger(__name__)
OVERSAMPLED_ENGINE = "full_row_adaptive_oversampling"


class OversampledStream(NamedTuple):
    """Prepared inputs of one adaptively oversampled expectation (order 1)."""

    coarse: FullRowStream  # pass 1: every coarse pose of the stage, at the pass-1 window
    # The child grid at the pass-2 window, parent-major: rows c R_c ... (c + 1) R_c - 1 are the children of coarse
    # rotation c, and translations t T_c ... (t + 1) T_c - 1 those of coarse translation t; parents and parent priors.
    fine: FullRowStream
    # The pass-2 window at one zero translation: its reader gives a tile's unshifted operands.
    base: FullRowStream
    rotation_children: int  # R_c
    translation_children: int  # T_c
    adaptive_fraction: float
    max_significant: int  # significant coarse samples kept per image at most (RELION's maxsig)
    # Jobs per pass-2 program at most (:func:`_chunk_sizes`); the moment programs take ``rotation_children`` rows
    # per job of a chunk.
    job_chunk: int
    fine_fraction: float  # the posterior mass an image's accumulated fine samples hold at least
    # (F_1,) int32: each pass-1 operand slot's position in a frame of the pass-2 operands (-1: GEMM padding).
    pass1_slots: jax.Array
    # Single particles: the phase factors of the coarse translations on the pass-1 window and of the child
    # translations on the pass-2 window (:func:`_shift_phases`); None for a reader whose factors depend on the tile.
    phases: tuple | None


def relion_child_grids(order: int, translations, translation_step: float):
    """RELION's order-1 children of the order-``order`` hidden rotation grid and of a translation grid.

    Returns the child rotations ``(8 R, 3, 3)`` and translations ``(T_c T, D)``, parent-major
    (:func:`relax.sampling.get_oversampled_rotation_grid_from_samples`,
    :func:`relax.sampling.get_oversampled_translation_grid`). A stage's grids are the same (read-only) arrays at
    every update.
    """
    translations = np.ascontiguousarray(translations, np.float32)
    return _child_grids(int(order), translations.tobytes(), translations.shape, float(translation_step))


@lru_cache(maxsize=8)
def _child_grids(order: int, translation_bytes: bytes, translation_shape, translation_step: float):
    translations = np.frombuffer(translation_bytes, np.float32).reshape(translation_shape)
    n_rotations = sampling.rotation_grid_size(order)
    rotations, rotation_parent = sampling.get_oversampled_rotation_grid_from_samples(
        np.arange(n_rotations), order, 1, rotation_index_order="relion"
    )[:2]
    children = rotations.shape[0] // n_rotations
    if not np.array_equal(rotation_parent, np.repeat(np.arange(n_rotations), children)):
        raise ValueError("Child rotations must be parent-major")
    fine_translations, translation_parent = sampling.get_oversampled_translation_grid(
        np.asarray(translations), float(translation_step), 1
    )
    grids = np.array(rotations, np.float32), np.array(fine_translations, np.float32)
    for grid in grids:
        grid.setflags(write=False)
    return grids


_DEVICE_ROTATIONS: dict = {}


def _device_rotations(rotations: np.ndarray, device) -> jax.Array:
    """The rotations with the row table's identity sentinel on ``device``. The upload is kept for the last grids
    used: a stage's child grid is the same array at every update (:func:`relion_child_grids`)."""
    key = (id(rotations), device)
    held = _DEVICE_ROTATIONS.get(key)
    if held is None or held[0] is not rotations:
        if len(_DEVICE_ROTATIONS) >= 4:
            _DEVICE_ROTATIONS.clear()
        with jax.default_device(device):
            uploaded = jnp.concatenate([jnp.asarray(rotations), jnp.eye(3, dtype=jnp.float32)[None]])
        _DEVICE_ROTATIONS[key] = held = (rotations, uploaded)
    return held[1]


def prepare_oversampled_stream(
    coarse: FullRowStream,
    fine_rotations,
    fine_translations,
    *,
    pass2: FullRowStream | None = None,
    adaptive_fraction: float = 0.999,
    max_significant: int = 100,
    job_chunk: int = 256,
    fine_fraction: float | None = None,
) -> OversampledStream:
    """The child grid of a pass-1 stream.

    ``fine_rotations`` ``(R_c R, 3, 3)`` and ``fine_translations`` ``(T_c T, D)`` are the children of the coarse
    stream's grids, parent-major (:func:`relion_child_grids`). A child takes its parent's rotation and translation
    log-priors. ``pass2`` is the same model, grids and priors at the pass-2 window (``coarse`` itself when both
    passes use one window); its model and windows are reused, not uploaded again. ``fine_fraction`` is the mass
    of each image's fine posterior that pass 2 accumulates (``adaptive_fraction`` when None, as RELION).
    """
    pass2 = coarse if pass2 is None else pass2
    if pass2.tile_loader is not None and not callable(getattr(pass2.tile_loader, "shift_phases", None)):
        raise ValueError("Adaptive oversampling needs a tile reader with shift_phases")
    fine_fraction = adaptive_fraction if fine_fraction is None else fine_fraction
    if not 0 < adaptive_fraction <= 1 or not 0 < fine_fraction <= 1:
        raise ValueError("The adaptive fractions must be in (0, 1]")
    n_rotations = int(coarse.arrays.rotations.shape[0]) - 1
    n_translations = int(coarse.translations.shape[0])
    fine_rotations = np.asarray(fine_rotations, np.float32)
    fine_translations = np.asarray(fine_translations, np.float32)
    if fine_rotations.shape[0] % n_rotations or fine_translations.shape[0] % n_translations:
        raise ValueError("Every coarse rotation and translation needs the same number of children")
    rotation_children = fine_rotations.shape[0] // n_rotations
    translation_children = fine_translations.shape[0] // n_translations
    if not 0 < max_significant <= n_rotations * n_translations:
        raise ValueError("max_significant must be between 1 and the number of coarse poses")
    rotation_parent = np.repeat(np.arange(n_rotations, dtype=np.int32), rotation_children)
    translation_parent = np.repeat(np.arange(n_translations, dtype=np.int32), translation_children)
    position = {int(pixel): slot for slot, pixel in enumerate(_window_pixel_ids(pass2)) if pixel >= 0}
    pass1_pixels = [int(pixel) for pixel in _window_pixel_ids(coarse)]
    if any(pixel >= 0 and pixel not in position for pixel in pass1_pixels):
        raise ValueError("The pass-1 window must lie inside the pass-2 window")
    arrays = pass2.arrays
    with jax.default_device(pass2.device):
        pass1_slots = jnp.asarray(np.asarray([position.get(pixel, -1) for pixel in pass1_pixels], np.int32))
        phases = None
        if pass2.tile_loader is None:
            phases = (_shift_phases(coarse, None, coarse.translations), _shift_phases(pass2, None, fine_translations))
        fine_arrays = arrays._replace(
            rotations=_device_rotations(fine_rotations, pass2.device),
            rotation_log_prior=jnp.append(jnp.repeat(arrays.rotation_log_prior[:-1], rotation_children), 0.0),
            translation_log_prior=jnp.repeat(arrays.translation_log_prior, translation_children),
            rotation_parent=jnp.asarray(np.append(rotation_parent, np.int32(n_rotations))),
            translation_parent=jnp.asarray(translation_parent),
            shift_squared=jnp.sum(jnp.asarray(fine_translations) ** 2, axis=-1),
        )
    fine = pass2._replace(
        arrays=fine_arrays,
        translations=fine_translations,
        n_coarse_rotations=n_rotations,
        n_coarse_translations=n_translations,
        block_starts=_block_starts(fine_rotations.shape[0], pass2.rotation_block_size, pass2.device),
        rotation_parent=rotation_parent,
    )
    # The pass-2 stream at one zero shift: its coarse grid keeps the reader's row table small.
    base = pass2._replace(translations=np.zeros((1, fine_translations.shape[1]), np.float32))
    return OversampledStream(
        coarse=coarse,
        fine=fine,
        base=base,
        rotation_children=int(rotation_children),
        translation_children=int(translation_children),
        adaptive_fraction=float(adaptive_fraction),
        max_significant=int(max_significant),
        job_chunk=int(job_chunk),
        fine_fraction=float(fine_fraction),
        pass1_slots=pass1_slots,
        phases=phases,
    )


def _window_pixel_ids(stream: FullRowStream) -> np.ndarray:
    """The packed half-image pixel of each operand slot of a frame (-1: a GPU stream's GEMM padding)."""
    arrays = stream.arrays
    return np.asarray(_window(arrays, stream.static) if arrays.gemm_window is None else arrays.gemm_window)


@partial(jax.jit, static_argnames=("n_frames", "shift_pad"))
def _pass1_operands(Y1, ctf2, phases, slots, *, n_frames, shift_pad):
    """The pass-1 score operands ``(K 2F_1, B T)`` and CTF rows ``(K F_1, B)`` from a tile's unshifted pass-2
    operands ``(K 2F, B)`` and ``(K F, B)``: the pixels at ``slots`` of each frame, times the coarse translations'
    phase factors ``(T, K, F_1)`` (zero at the padding slots), in the readers' layout (image major, shift minor)."""
    B, K = ctf2.shape[1], n_frames
    F = ctf2.shape[0] // K
    T = phases.shape[0]
    inside = slots >= 0
    take = jnp.maximum(slots, 0)
    planar = jnp.where(inside, Y1[:, :B].T.reshape(B, K, 2, F)[..., take], 0)  # (B, K, 2, F_1)
    shifts = jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32), (B, T))
    shifted = _shifted(planar.reshape(B, -1), phases, shifts, K)  # (B, T, K 2F_1)
    pass1_ctf2 = jnp.where(inside[None, :, None], ctf2.reshape(K, F, B)[:, take], 0)
    return jnp.pad(shifted.reshape(B * T, -1).T, ((0, 0), (0, shift_pad))), pass1_ctf2.reshape(-1, B)


def _pass1_tile(ostream: OversampledStream, tile, layout, significant_rows):
    """Pass 1's tile and layout from the tile's unshifted pass-2 operands: no second read of the images.

    Pass 1 reads the score operands only; the tile carries no reconstruction operands."""
    coarse = ostream.coarse
    coarse_mask, table, coarse_layout = tile_support(coarse, significant_rows)
    B = int(tile.y_norm.shape[0])
    # A reader's padding images (after the real ones) take the first image's support, as the reader gives them.
    coarse_mask = jnp.concatenate([coarse_mask, jnp.repeat(coarse_mask[:1], B - coarse_mask.shape[0], axis=0)])
    T = int(coarse.translations.shape[0])
    phases = _shift_phases(coarse, layout, coarse.translations) if ostream.phases is None else ostream.phases[0]
    Y1, ctf2 = _pass1_operands(
        tile.Y1,
        tile.ctf2,
        phases,
        ostream.pass1_slots,
        n_frames=_n_frames(tile),
        shift_pad=-(B * T) % _SHIFT_ALIGN if coarse.static.cuda_kernels else 0,
    )
    pass1 = tile._replace(coarse_mask=coarse_mask, rows=table, Y1=Y1, ctf2=ctf2, Y1_recon=None, ctf2_recon=None)
    if "n_real" in layout:
        coarse_layout["n_real"] = layout["n_real"]
    return pass1, coarse_layout


@partial(jax.jit, static_argnames=("n_significant", "every"))
def _significant_samples(score, rows, posterior, fraction, n_real, *, n_significant, every):
    """Each image's significant coarse samples ``(B, S)``: rotation, translation and a validity mask; and per image
    the posterior mass they hold and whether the cap stopped them before ``fraction``.

    The largest posterior weights are kept until their running sum exceeds ``fraction`` of the image's mass (the
    weights are normalized), at most ``S`` of them; every kept weight is at least the last one counted, and positive.
    ``every`` (fraction 1) keeps every supported pose among the ``S`` largest. Images from ``n_real`` on (a reader's
    padding) keep none.
    """
    capacity, B, T = score.shape
    # The S largest samples of an image lie in its S rows of largest maximum: select among those rows only.
    n_rows = min(n_significant, capacity)
    _, row = jax.lax.top_k(jnp.max(score, axis=2).T, n_rows)  # (B, n_rows)
    candidates = score[row, jnp.arange(B)[:, None]]  # (B, n_rows, T)
    top, where = jax.lax.top_k(candidates.reshape(B, n_rows * T), n_significant)
    position = jnp.take_along_axis(row, where // T, axis=1) * T + where % T
    weight = jnp.exp((top - posterior.center[:, None]) - posterior.centered_logZ[:, None])
    if every:
        valid = jnp.isfinite(top)  # every supported pose, also those whose float32 weight underflows
        capped = jnp.zeros((B,), bool)
    else:
        crossed = jnp.cumsum(weight, axis=1) > fraction
        capped = ~jnp.any(crossed, axis=1)
        count = jnp.where(capped, n_significant, jnp.argmax(crossed, axis=1) + 1)
        threshold = jnp.take_along_axis(weight, (count - 1)[:, None], axis=1)
        valid = (weight >= threshold) & (weight > 0)
    real = jnp.arange(B) < n_real
    valid = valid & real[:, None]
    mass = jnp.sum(jnp.where(valid, weight, 0), axis=1)
    return rows[position // T], (position % T).astype(jnp.int32), valid, mass, capped & real


class _FineJobs(NamedTuple):
    """Pass-2 jobs: the slots ``b S + s`` of a tile's image-major significant samples, and one more that the
    padding of a chunk addresses (invalid)."""

    image: jax.Array  # (J,) int32 tile image
    rotation: jax.Array  # (J,) int32 coarse rotation
    translation: jax.Array  # (J,) int32 coarse translation
    valid: jax.Array  # (J,) bool


class _JobResults(NamedTuple):
    """Pass-2 results of every job slot (:class:`_FineJobs`); unscored slots keep a score of ``-inf``."""

    score: jax.Array  # (J, R_c, T_c)
    mean: jax.Array  # (J, q, R_c, T_c)
    covariance: jax.Array  # (J, R_c, tri(q))


@jax.jit
def _job_table(rotation, translation, valid):
    """The jobs of every slot of the significant samples ``(B, S)``, and the padding slot ``B S``."""
    B, S = valid.shape
    slot = jnp.arange(B * S + 1, dtype=jnp.int32)
    return _FineJobs(
        image=jnp.minimum(slot // S, B - 1),
        rotation=jnp.append(rotation.reshape(-1), 0).astype(jnp.int32),
        translation=jnp.append(translation.reshape(-1), 0).astype(jnp.int32),
        valid=jnp.append(valid.reshape(-1), False),
    )


def _empty_results(n_jobs: int, rotation_children: int, translation_children: int, q: int) -> _JobResults:
    return _JobResults(
        score=jnp.full((n_jobs, rotation_children, translation_children), -jnp.inf, jnp.float32),
        mean=jnp.zeros((n_jobs, q, rotation_children, translation_children), jnp.float32),
        covariance=jnp.zeros((n_jobs, rotation_children, tri_size(q)), jnp.float32),
    )


class _FineRows(NamedTuple):
    """Pass-2 rows: one image at one child rotation of one of its jobs, with the job's child translations."""

    image: jax.Array  # (N,) int32 tile image
    row: jax.Array  # (N,) int32 child rotation
    parent: jax.Array  # (N,) int32 coarse rotation
    translation: jax.Array  # (N,) int32 coarse translation
    valid: jax.Array  # (N,) bool


def _job_poses(arrays, jobs, rotation_children, translation_children):
    """Child rows ``(J, R_c)``, child translations ``(J, T_c)`` and log-priors ``(J, R_c, T_c)``."""
    rows = jobs.rotation[:, None] * rotation_children + jnp.arange(rotation_children, dtype=jnp.int32)
    shifts = jobs.translation[:, None] * translation_children + jnp.arange(translation_children, dtype=jnp.int32)
    prior = arrays.rotation_log_prior[rows][:, :, None] + arrays.translation_log_prior[shifts][:, None, :]
    prior = jnp.where(jobs.valid[:, None, None], prior, -jnp.inf)
    return rows, shifts, prior


def _shifted(planar, phases, shifts, n_frames):
    """Planar operands ``(J, T_c, K 2F)`` of each job's image (``planar`` ``(J, K 2F)``, frame by frame
    ``[Re | Im]``) at its child translations: the complex image times each child's phase factor ``(T, K, F)``, the
    product the tile readers form for a shifted operand."""
    J = planar.shape[0]
    planar = planar.reshape(J, n_frames, 2, -1)
    image = jax.lax.complex(planar[:, :, 0], planar[:, :, 1])  # (J, K, F)
    shifted = image[:, None] * phases[shifts]  # (J, T_c, K, F)
    return _real_imag(shifted).reshape(J, shifts.shape[1], -1)


def _job_projections(arrays, tile, rows, static):
    """Planar windowed projections ``(P, J, R_c, K 2F)`` of the job children, frame by frame ``[Re | Im]`` (the
    score GEMM rows of :func:`_latent_block`), and their packed pair products ``(tri(P), J, R_c, K F)``: the
    relax CUDA projector on GPU streams (window padded as the tile operands), else recovar's projection."""
    J, R_c = rows.shape
    rotations = _frame_rotations(tile, arrays.rotations[rows.reshape(-1)])
    if static.cuda_kernels:
        planar, products = ppca_window_project_f32(
            arrays.augmented_voxel_major,
            arrays.gemm_window,
            rotations,
            image_shape=static.image_shape,
            volume_shape=static.volume_shape,
            max_r=static.projection_max_r,
            with_products=True,
        )
        return planar.reshape(planar.shape[0], J, R_c, -1), products.reshape(products.shape[0], J, R_c, -1)
    proj = _project(arrays, rotations, static)  # (P, J R_c K, F)
    P = proj.shape[0]
    proj = proj.reshape(P, J * R_c, -1, proj.shape[-1])
    first, second = np.triu_indices(P)
    real, imag = proj.real, proj.imag
    products = jnp.stack([real[i] * real[j] + imag[i] * imag[j] for i, j in zip(first.tolist(), second.tolist())])
    return _real_imag(proj).reshape(P, J, R_c, -1), products.reshape(first.size, J, R_c, -1)


@partial(jax.jit, static_argnames=("static", "rotation_children", "translation_children", "chunk"), donate_argnums=(0,))
def _score_jobs(
    results, arrays, tile, phases, table, order, start, *, static, rotation_children, translation_children, chunk
):
    """Pass 2 scores of the ``chunk`` jobs ``order[start:]`` (slots of ``table``) at their children, written at
    their slots of ``results``: scores ``(R_c, T_c)``, latent means ``(q, R_c, T_c)`` and packed covariances
    ``(R_c, tri(q))`` per job, the dense stream's (:func:`_latent_block`) for one image, its child rotations and
    child translations. The padding of ``order`` addresses the invalid last slot, whose score stays ``-inf``."""
    slots = jax.lax.dynamic_slice_in_dim(order, start, chunk)
    jobs = _FineJobs(*(x[slots] for x in table))
    rows, shifts, prior = _job_poses(arrays, jobs, rotation_children, translation_children)
    planar, products = _job_projections(arrays, tile, rows, static)
    Y1 = _shifted(tile.Y1[:, jobs.image].T, phases, shifts, _n_frames(tile))  # (J, T_c, K 2F)
    inner = jnp.einsum("pjrk,jtk->jprt", planar, Y1, precision=_gemm(static))
    gram = jnp.einsum("mjrk,kj->jmr", products, tile.ctf2[:, jobs.image], precision=_gemm(static))

    def one(inner, gram, prior):
        score, mean, covariance = _latent_scores(inner[:, :, None], gram[:, :, None], prior[None])
        return score[:, 0], mean[:, :, 0], covariance[:, 0]

    score, mean, covariance = jax.vmap(one)(inner, gram, prior)
    return _JobResults(
        score=results.score.at[slots].set(score),
        mean=results.mean.at[slots].set(mean),
        covariance=results.covariance.at[slots].set(covariance),
    )


@jax.jit
def _significant_rows(score, table, posterior, threshold):
    """Whether each job's child rotations ``(J, R_c)`` hold a significant fine sample of the job's image: a
    posterior weight of at least the image's ``threshold``, and positive."""
    center = posterior.center[table.image][:, None, None]
    weight = jnp.exp((score - center) - posterior.centered_logZ[table.image][:, None, None])
    return table.valid[:, None] & jnp.any((weight >= threshold[table.image][:, None, None]) & (weight > 0), axis=2)


@partial(jax.jit, static_argnames=("static", "translation_children", "chunk", "moments"), donate_argnums=(0,))
def _moment_rows(
    carry,
    arrays,
    tile,
    phases,
    jobs,
    results,
    posterior,
    threshold,
    take,
    n_rows,
    start,
    *,
    static,
    translation_children,
    chunk,
    moments,
):
    """Pass 2 posterior, M-step images and diagnostics of ``chunk`` kept rows from ``start``, added into ``carry``.

    ``take`` lists the kept rows ``job R_c + child`` of the job ``results``, the first ``n_rows`` of it. A row's weights are
    its image's posterior at the row's child translations, zero below the image's fine significance ``threshold``.
    ``carry.rotation_mass`` is per coarse rotation: each child's posterior mass is summed into its parent.
    """
    N, R_c = chunk, results.score.shape[1]
    position = start + jnp.arange(N, dtype=jnp.int32)
    index = jax.lax.dynamic_slice_in_dim(take, start, N)
    job, child = index // R_c, index % R_c
    rows = _FineRows(
        image=jobs.image[job],
        row=jobs.rotation[job] * R_c + child,
        parent=jobs.rotation[job],
        translation=jobs.translation[job],
        valid=position < n_rows,
    )
    score, mean, covariance = results.score[job, child], results.mean[job, :, child], results.covariance[job, child]
    shifts = rows.translation[:, None] * translation_children + jnp.arange(translation_children, dtype=jnp.int32)
    centered = (score - posterior.center[rows.image][:, None]) - posterior.centered_logZ[rows.image][:, None]
    weight = jnp.exp(centered)  # (N, T_c)
    significant = rows.valid[:, None] & (weight >= threshold[rows.image][:, None]) & (weight > 0)
    gamma = jnp.where(significant, weight, 0)
    q = mean.shape[1]
    diagonal = [k for k, (i, j) in enumerate(zip(*np.triu_indices(q))) if i == j]
    latent_trace = sum((covariance[..., k] for k in diagonal), jnp.zeros(covariance.shape[:1], covariance.dtype))
    row_mass = jnp.sum(gamma, axis=1)  # (N,)
    carry = carry._replace(
        embedding=carry.embedding.at[rows.image].add(jnp.einsum("nt,nqt->nq", gamma, mean, precision=_HIGHEST)),
        rotation_mass=carry.rotation_mass.at[rows.parent].add(row_mass),
        latent_covariance_trace_sum=carry.latent_covariance_trace_sum + jnp.sum(row_mass * latent_trace),
        pose_entropy_sum=carry.pose_entropy_sum - jnp.sum(jnp.where(significant, gamma * centered, 0)),
        offset_second_sum=carry.offset_second_sum + jnp.sum(gamma * arrays.shift_squared[shifts]),
        n_significant=carry.n_significant.at[rows.image].add(jnp.sum(gamma > 1e-3, axis=1).astype(jnp.int32)),
    )
    if not moments:
        return carry
    P = static.basis_size
    K = _n_frames(tile)
    F = tile.ctf2_recon.shape[1] // K
    latent = jnp.moveaxis(mean, 1, 0)  # (q, N, T_c)
    weights = jnp.concatenate([gamma[None], gamma[None] * latent], axis=0)  # (P, N, T_c)
    # (tri(P), N, 1): the moment sums over each row's child translations.
    sums = _second_moment_sums(gamma[:, None, :], latent[:, :, None, :], covariance[:, None, :])
    recon = _shifted(tile.Y1_recon[rows.image], phases, shifts, K)  # (N, T_c, K 2F)
    rhs_parts = jnp.einsum("pnt,ntk->pnk", weights, recon, precision=_gemm(static)).reshape(P, N * K, 2 * F)
    # No contraction: full float32 products cost nothing here, and TF32 would round each moment
    # entry on its own and leave the metric indefinite (full_row_stream._metric_dot).
    lhs_images = jnp.einsum("mn,nk->mnk", sums[:, :, 0], tile.ctf2_recon[rows.image], precision=_HIGHEST).reshape(
        sums.shape[0], N * K, F
    )
    rotations = _frame_rotations(tile, arrays.rotations[rows.row])
    return _scatter_moment_images(carry, arrays, static, rhs_parts, lhs_images, rotations)


# A tile's jobs (and rows) run in chunks of the planned size and of 1/4 and 1/16 of it: whole large chunks first,
# the remainder in the smallest, so a tile runs a few programs and pads less than one smallest chunk.
CHUNK_RATIO = 4
CHUNK_LEVELS = 3


def _chunk_sizes(largest: int) -> tuple[int, ...]:
    return tuple(dict.fromkeys(max(1, int(largest) // CHUNK_RATIO**level) for level in range(CHUNK_LEVELS)))


def _chunks(mask, sizes, padding: int):
    """The set positions of the flattened ``mask``, followed by ``padding`` up to its size plus the largest of the
    chunk ``sizes`` (so a program that slices a chunk of them has one shape per chunk size and mask size), how many
    are set, and the ``(start, size)`` chunks that cover them (:func:`_chunk_sizes`)."""
    mask = np.asarray(mask).reshape(-1)
    kept = np.flatnonzero(mask).astype(np.int32)
    order = np.full(mask.size + sizes[0], padding, np.int32)
    order[: kept.size] = kept
    plan, start = [], 0
    for level, size in enumerate(sizes):
        last = level == len(sizes) - 1
        while kept.size - start >= size or (last and start < kept.size):
            plan.append((start, size))
            start += size
    return jnp.asarray(order), int(kept.size), plan


@partial(jax.jit, static_argnames=("every",))
def _fine_posterior(score, rotation, translation, fraction, *, every):
    """The fine posterior of a tile from its job scores ``(B S + 1, R_c, T_c)`` (image-major slots and the padding
    slot) and the samples' coarse rotations and translations ``(B, S)``: per-image normalization, the first
    maximum's child pose (slot, rotation, translation order), and each image's fine significance threshold, the
    smallest of its largest posterior weights whose running sum exceeds ``fraction`` (RELION's rule, as
    :func:`_significant_samples` without a cap); 0 with ``every`` (fraction 1)."""
    B, S = rotation.shape
    R_c, T_c = score.shape[1:]
    per_image = score[:-1].reshape(B, -1)
    center = jnp.max(per_image, axis=1)
    centered_logZ = jnp.log(jnp.sum(jnp.exp(per_image - center[:, None]), axis=1))
    first = jnp.argmax(per_image, axis=1)
    sample, child = first // (R_c * T_c), first % (R_c * T_c)
    top_rotation = jnp.take_along_axis(rotation, sample[:, None], axis=1)[:, 0] * R_c + child // T_c
    top_translation = jnp.take_along_axis(translation, sample[:, None], axis=1)[:, 0] * T_c + child % T_c
    posterior = _Posterior(
        center=center,
        centered_logZ=centered_logZ,
        top_score=center,
        top_rotation=top_rotation.astype(jnp.int32),
        top_translation=top_translation.astype(jnp.int32),
    )
    if every:
        return posterior, jnp.zeros((B,), score.dtype)
    weight = -jnp.sort(-jnp.exp((per_image - center[:, None]) - centered_logZ[:, None]), axis=1)
    crossed = jnp.cumsum(weight, axis=1) > fraction
    count = jnp.where(jnp.any(crossed, axis=1), jnp.argmax(crossed, axis=1) + 1, weight.shape[1])
    return posterior, jnp.take_along_axis(weight, (count - 1)[:, None], axis=1)[:, 0]


def _significant(ostream: OversampledStream, coarse_tile, coarse_layout, kept, posterior):
    n_real = _n_real(coarse_tile, coarse_layout)
    fraction = ostream.adaptive_fraction
    return _significant_samples(
        kept.score,
        coarse_tile.rows,
        posterior,
        jnp.float32(fraction),
        np.int32(n_real),
        n_significant=ostream.max_significant,
        every=fraction >= 1.0,
    )


def _shift_phases(base: FullRowStream, layout, translations):
    """Phase factors ``(T, K, F)`` of ``translations`` on the stream's window, from the tile's reader (the
    single-particle reader's :func:`relax.fourier.preprocessing.half_translation_phase_table` on the score window,
    which no tile enters), padded with ones to the GEMM window on GPU streams (the operands are zero there)."""
    translations = np.ascontiguousarray(translations, np.float32)
    pad = None if not base.static.cuda_kernels else int(base.arrays.gemm_window.shape[0])
    if base.tile_loader is None:
        window = np.ascontiguousarray(_window(base.arrays, base.static), np.int32)
        return _single_particle_phases(
            translations.tobytes(), translations.shape, base.static.image_shape, window.tobytes(), pad, base.device
        )
    phases = base.tile_loader.shift_phases(base, layout, translations)
    return phases if pad is None else _pad_phases(phases, pad)


def _pad_phases(phases, n_slots: int):
    return jnp.pad(phases, ((0, 0), (0, 0), (0, n_slots - int(phases.shape[-1]))), constant_values=1)


@lru_cache(maxsize=16)
def _single_particle_phases(translation_bytes, translation_shape, image_shape, window_bytes, pad, device):
    """Single-particle phase factors ``(T, 1, F)``: they depend on the grids and the window alone, so a stage
    forms them once."""
    translations = np.frombuffer(translation_bytes, np.float32).reshape(translation_shape)
    window = np.frombuffer(window_bytes, np.int32)
    with jax.default_device(device):
        phases = half_translation_phase_table(translations, image_shape)[:, window][:, None, :]
        return phases if pad is None else _pad_phases(phases, pad)


def _oversampled_tile(ostream: OversampledStream, image_indices, significant_rows, *, collect_observation, moments):
    """Both passes of one tile: the unshifted fine tile, its layout, the fine posterior, the carry and the
    significant samples ``(rotation, translation, valid, mass, capped)``."""
    coarse, fine, base = ostream.coarse, ostream.fine, ostream.base
    tile, observation_power, layout = _read_tile(
        base, image_indices, [None] * len(image_indices), collect_observation=collect_observation
    )
    coarse_tile, coarse_layout = _pass1_tile(ostream, tile, layout, significant_rows)
    kept, posterior = _score_tile(coarse, coarse_tile, coarse_layout["n_blocks"])
    n_real = _n_real(coarse_tile, coarse_layout)
    _check_finite_posterior(posterior, n_real)
    significant = _significant(ostream, coarse_tile, coarse_layout, kept, posterior)
    rotation, translation, valid = significant[:3]
    del kept, coarse_tile
    phases = _shift_phases(base, layout, fine.translations) if ostream.phases is None else ostream.phases[1]
    if ostream.job_chunk > launch_job_chunk(ostream):
        raise ValueError(
            f"A chunk of {ostream.job_chunk} jobs exceeds one kernel launch ({launch_job_chunk(ostream)} jobs at most)"
        )
    B, S = valid.shape
    R_c, T_c = ostream.rotation_children, ostream.translation_children
    table = _job_table(rotation, translation, valid)
    sizes = _chunk_sizes(ostream.job_chunk)
    order, _, chunks = _chunks(jax.device_get(valid), sizes, padding=B * S)
    results = _empty_results(B * S + 1, R_c, T_c, fine.static.basis_size - 1)
    for start, size in chunks:
        results = _score_jobs(
            results,
            fine.arrays,
            tile,
            phases,
            table,
            order,
            np.int32(start),
            static=fine.static,
            rotation_children=R_c,
            translation_children=T_c,
            chunk=size,
        )
    fine_fraction = ostream.fine_fraction
    fine_posterior, threshold = _fine_posterior(
        results.score, rotation, translation, jnp.float32(fine_fraction), every=fine_fraction >= 1.0
    )
    _check_finite_posterior(fine_posterior, n_real)
    fine_posterior = _drop_padding(fine_posterior, n_real)
    n_coarse = int(coarse.arrays.rotations.shape[0]) - 1
    power = jnp.float32(0) if observation_power is None else observation_power
    carry = _empty_carry(fine, B, power)._replace(rotation_mass=jnp.zeros((n_coarse,), jnp.float32))
    row_mask = _significant_rows(results.score, table, fine_posterior, threshold)
    take, n_rows, chunks = _chunks(jax.device_get(row_mask), tuple(size * R_c for size in sizes), padding=0)
    for start, size in chunks:
        carry = _moment_rows(
            carry,
            fine.arrays,
            tile,
            phases,
            table,
            results,
            fine_posterior,
            threshold,
            take,
            np.int32(n_rows),
            np.int32(start),
            static=fine.static,
            translation_children=T_c,
            chunk=size,
            moments=moments,
        )
    layout = dict(
        layout,
        scored_rows=coarse_layout["scored_rows"],
        supported_image_rows=coarse_layout["supported_image_rows"],
        pass2_rows=n_rows,
    )
    return tile, layout, fine_posterior, carry, significant


@full_float32
def accumulate_oversampled_tiles(ostream: OversampledStream, tiles, *, enforce_x0=True):
    """:func:`accumulate_oversampled_tile` for each ``(image_indices, significant_rows)`` in ``tiles``."""
    return [accumulate_oversampled_tile(ostream, ids, rows, enforce_x0=enforce_x0) for ids, rows in tiles]


@full_float32
def accumulate_oversampled_tile(ostream: OversampledStream, image_indices, significant_rows=None, *, enforce_x0=True):
    """Unregularized PPCA statistics of one tile with adaptive oversampling.

    The same statistics and diagnostics as :func:`relax.ppca_refinement.full_row_stream.accumulate_full_row_tile`,
    over each image's pass-2 poses; ``rotation_mass`` is per coarse rotation (children summed into their parent).
    The diagnostics add, per image, the packed coarse ids ``rotation T + translation`` of its significant samples
    (``significant_samples``), their count, the posterior mass they hold and whether ``max_significant`` stopped
    them before the adaptive fraction (``significant_capped_per_image``); and the tile's accumulated (image, child
    rotation) rows (``pass2_rows``) of the ``scored_fine_rows`` its jobs scored.
    """
    significant_rows = [None] * len(image_indices) if significant_rows is None else significant_rows
    coarse = ostream.coarse
    with jax.default_device(coarse.device):
        tile, layout, posterior, carry, significant = _oversampled_tile(
            ostream, image_indices, significant_rows, collect_observation=True, moments=True
        )
        n_coarse = int(carry.rotation_mass.shape[0])
        host = jax.device_get(dict(zip(("rotation", "translation", "valid", "mass", "capped"), significant)))
        n_real = _n_real(tile, layout)
        packed = host["rotation"] * int(coarse.translations.shape[0]) + host["translation"]
        samples = [np.sort(packed[i][host["valid"][i]]).astype(np.int64) for i in range(n_real)]
        layout = dict(layout, rows=np.arange(n_coarse, dtype=np.int32))
        stats = _finish_full_row_tile(coarse, image_indices, tile, layout, posterior, carry, enforce_x0=enforce_x0)
    stats.diagnostics.update(
        {
            "engine": OVERSAMPLED_ENGINE,
            "adaptive_fraction": ostream.adaptive_fraction,
            "max_significant": ostream.max_significant,
            "fine_fraction": ostream.fine_fraction,
            "scored_fine_rows": int(sum(x.size for x in samples)) * ostream.rotation_children,
            "significant_samples": samples,
            "significant_samples_per_image": np.asarray([x.size for x in samples], np.int32),
            "significant_mass_per_image": np.asarray(host["mass"][:n_real], np.float32),
            "significant_capped_per_image": np.asarray(host["capped"][:n_real], bool),
        }
    )
    return stats


@full_float32
def oversampled_tile_embeddings(
    ostream: OversampledStream, image_indices, significant_rows=None
) -> DensePPCAEmbeddings:
    """Pose-marginal embeddings of one tile with adaptive oversampling, as
    :func:`relax.ppca_refinement.full_row_stream.full_row_tile_embeddings`."""
    significant_rows = [None] * len(image_indices) if significant_rows is None else significant_rows
    with jax.default_device(ostream.coarse.device):
        tile, layout, _, carry, _ = _oversampled_tile(
            ostream, image_indices, significant_rows, collect_observation=False, moments=False
        )
        n_real = _n_real(tile, layout)
        return DensePPCAEmbeddings(carry.embedding[:n_real], layout["original_ids"], n_real)


_JOB_PROGRAM_BYTES: dict = {}


def job_program_bytes(ostream: OversampledStream, n_images: int, chunk: int) -> int:
    """Device bytes of pass 2 for chunks of ``chunk`` jobs in a tile of ``n_images``: the job results of every
    slot (:class:`_JobResults`) and the larger of the programs :func:`_score_jobs` and :func:`_moment_rows`
    (temporaries plus outputs that do not alias a donated input), from XLA's compiled memory analysis at those
    shapes; cached per stage shape."""
    fine, base = ostream.fine, ostream.base
    n_jobs = int(n_images) * ostream.max_significant + 1
    key = (_plan_shape_key(fine), int(n_images), n_jobs, int(chunk), getattr(fine.device, "id", None))
    if key not in _JOB_PROGRAM_BYTES:
        spec = jax.ShapeDtypeStruct
        R_c, T_c = ostream.rotation_children, ostream.translation_children
        tile = _tile_spec(base, n_images)
        K = 1 if tile.frames is None else tile.frames.shape[0]
        F = tile.ctf2.shape[0] // K
        phases = spec((int(fine.translations.shape[0]), K, F), jnp.complex64)
        table = _FineJobs(*(spec((n_jobs,), jnp.int32),) * 3, spec((n_jobs,), jnp.bool_))
        q = fine.static.basis_size - 1
        results = jax.eval_shape(lambda: _empty_results(n_jobs, R_c, T_c, q))
        rows = chunk * R_c  # one moment program's rows: a job chunk's child rotations
        n_coarse = int(ostream.coarse.arrays.rotations.shape[0]) - 1
        carry = jax.eval_shape(
            lambda: _empty_carry(fine, int(n_images), jnp.float32(0))._replace(
                rotation_mass=jnp.zeros((n_coarse,), jnp.float32)
            )
        )
        vector, index = spec((int(n_images),), jnp.float32), spec((int(n_images),), jnp.int32)
        scalar = spec((), jnp.int32)
        posterior = _Posterior(vector, vector, vector, index, index)
        with jax.default_device(fine.device):
            programs = (
                _score_jobs.lower(
                    results,
                    fine.arrays,
                    tile,
                    phases,
                    table,
                    spec((n_jobs - 1 + chunk,), jnp.int32),
                    scalar,
                    static=fine.static,
                    rotation_children=R_c,
                    translation_children=T_c,
                    chunk=chunk,
                ),
                _moment_rows.lower(
                    carry,
                    fine.arrays,
                    tile,
                    phases,
                    table,
                    results,
                    posterior,
                    vector,
                    spec((n_jobs * R_c + rows,), jnp.int32),
                    scalar,
                    scalar,
                    static=fine.static,
                    translation_children=T_c,
                    chunk=rows,
                    moments=True,
                ),
            )
            analyses = [program.compile().memory_analysis() for program in programs]
        held = sum(int(np.prod(x.shape)) * x.dtype.itemsize for x in results)
        _JOB_PROGRAM_BYTES[key] = held + max(
            int(a.temp_size_in_bytes) + int(a.output_size_in_bytes) - int(a.alias_size_in_bytes) for a in analyses
        )
    return _JOB_PROGRAM_BYTES[key]


# Jobs per pass-2 program at most; the plan halves it until the programs fit the device.
MAX_JOB_CHUNK = 1024
# Rotations one launch of the CUDA window projector and moment scatter takes (their grids are one block row per
# rotation; CUDA's grid limit). A job chunk projects ``chunk * rotation_children * frames`` rotations.
KERNEL_LAUNCH_ROTATIONS = 65535


def launch_job_chunk(ostream: OversampledStream) -> int:
    """The largest power-of-two job chunk, at most :data:`MAX_JOB_CHUNK`, whose child rotations over the tile's
    frames one kernel launch takes (:data:`KERNEL_LAUNCH_ROTATIONS`); CPU streams have no such limit."""
    chunk = MAX_JOB_CHUNK
    if ostream.fine.static.cuda_kernels:
        per_job = ostream.rotation_children * _tile_frames(ostream.base)
        if per_job > KERNEL_LAUNCH_ROTATIONS:
            raise ValueError(f"One job's {per_job} child rotations over its frames exceed one kernel launch")
        while chunk * per_job > KERNEL_LAUNCH_ROTATIONS:
            chunk //= 2
    return chunk


_JOB_CHUNKS: dict = {}


def plan_job_chunk(ostream: OversampledStream, n_images: int, *, memory_bytes=None, device_bytes=None) -> int:
    """Jobs per pass-2 program: the most one kernel launch takes (:func:`launch_job_chunk`), halved until the job
    programs (:func:`job_program_bytes`) fit the device's free memory less the planner's fragmentation headroom
    (:func:`relax.ppca_refinement.full_row_stream.plan_tile_images`). Pass 1's buffers are released before pass 2,
    so the job programs have the device to themselves, apart from the tile's unshifted operands and job results."""
    if memory_bytes is None and device_bytes is None:
        # The device probes run nvidia-smi (tens of ms); a stage's shapes are planned once.
        key = (_plan_shape_key(ostream.fine), int(n_images), getattr(ostream.fine.device, "id", None))
        if key not in _JOB_CHUNKS:
            _JOB_CHUNKS[key] = plan_job_chunk(
                ostream,
                n_images,
                memory_bytes=_available_bytes(ostream.fine)[0],
                device_bytes=_available_bytes(ostream.fine)[1],
            )
        return _JOB_CHUNKS[key]
    budget = memory_bytes - TILE_FRAGMENTATION_HEADROOM * (device_bytes or memory_bytes)
    chunk = launch_job_chunk(ostream)
    while chunk > 1 and job_program_bytes(ostream, n_images, chunk) > budget:
        chunk //= 2
    counted = job_program_bytes(ostream, n_images, chunk)
    if counted > budget:
        raise ValueError(
            f"One pass-2 job needs {counted / 2**30:.2f} GiB, more than the {budget / 2**30:.2f} GiB budget: "
            "use a smaller image batch or rotation grid"
        )
    logger.info(
        "PPCA oversampling plan: %d of %d jobs per pass-2 program; counted %.2f GiB of a %.2f GiB budget",
        chunk,
        MAX_JOB_CHUNK,
        counted / 2**30,
        budget / 2**30,
    )
    return chunk


def significance_summary(counts, mass, capped, *, low_mass: float) -> dict:
    """Summary of a batch's significant samples: their count per image (median, mean), the share of images the cap
    stopped short of the adaptive fraction, the posterior mass those images hold (mean, 5th percentile, minimum)
    and the share of the batch both capped and holding less than ``low_mass``."""
    counts, mass, capped = np.asarray(counts), np.asarray(mass, np.float64), np.asarray(capped, bool)
    held = mass[capped]
    return {
        "samples_median": float(np.median(counts)),
        "samples_mean": float(np.mean(counts)),
        "capped_fraction": float(np.mean(capped)),
        "capped_mass_mean": float(np.mean(held)) if held.size else None,
        "capped_mass_p5": float(np.percentile(held, 5)) if held.size else None,
        "capped_mass_min": float(np.min(held)) if held.size else None,
        "capped_low_mass_fraction": float(np.mean(capped & (mass < low_mass))),
    }


__all__ = [
    "OVERSAMPLED_ENGINE",
    "OversampledStream",
    "accumulate_oversampled_tile",
    "accumulate_oversampled_tiles",
    "job_program_bytes",
    "oversampled_tile_embeddings",
    "plan_job_chunk",
    "prepare_oversampled_stream",
    "relion_child_grids",
    "significance_summary",
]
