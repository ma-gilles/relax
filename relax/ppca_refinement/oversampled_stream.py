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

Each significant sample is one job: an image, a coarse rotation and a coarse translation, scored at its children.
Jobs are laid out image-major, ``max_significant`` per image (unused slots are invalid), and run in fixed-size chunks,
so every program keeps its shapes.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from relax import sampling
from relax.cuda.kernels import ppca_window_project_f32
from relax.ppca_refinement.dense_dataset import DensePPCAEmbeddings
from relax.ppca_refinement.full_row_stream import (
    _HIGHEST,
    FullRowStream,
    _block_starts,
    _check_finite_posterior,
    _drop_padding,
    _empty_carry,
    _finish_full_row_tile,
    _frame_rotations,
    _gemm,
    _latent_scores,
    _n_real,
    _Posterior,
    _project,
    _read_tile,
    _real_imag,
    _scatter_moment_images,
    _score_tile,
    _second_moment_sums,
)
from relax.ppca_refinement.residual_statistics import full_float32

OVERSAMPLED_ENGINE = "full_row_adaptive_oversampling"


class OversampledStream(NamedTuple):
    """Prepared inputs of one adaptively oversampled expectation (order 1)."""

    coarse: FullRowStream  # pass 1: every coarse pose of the stage
    # The child grid, parent-major: rows c R_c ... (c + 1) R_c - 1 are the children of coarse rotation c, and
    # translations t T_c ... (t + 1) T_c - 1 those of coarse translation t; parents and parent priors as in pass 1.
    fine: FullRowStream
    rotation_children: int  # R_c
    translation_children: int  # T_c
    adaptive_fraction: float
    max_significant: int  # significant coarse samples kept per image at most (RELION's maxsig)
    job_chunk: int  # jobs per pass-2 program


def relion_child_grids(order: int, translations, translation_step: float):
    """RELION's order-1 children of the order-``order`` hidden rotation grid and of a translation grid.

    Returns the child rotations ``(8 R, 3, 3)`` and translations ``(T_c T, D)``, parent-major
    (:func:`relax.sampling.get_oversampled_rotation_grid_from_samples`,
    :func:`relax.sampling.get_oversampled_translation_grid`).
    """
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
    return np.asarray(rotations, np.float32), np.asarray(fine_translations, np.float32)


def prepare_oversampled_stream(
    coarse: FullRowStream,
    fine_rotations,
    fine_translations,
    *,
    adaptive_fraction: float = 0.999,
    max_significant: int = 100,
    job_chunk: int = 256,
) -> OversampledStream:
    """Add the child grid to a pass-1 stream; the model and windows are shared, not uploaded again.

    ``fine_rotations`` ``(R_c R, 3, 3)`` and ``fine_translations`` ``(T_c T, D)`` are the children of the coarse
    stream's grids, parent-major (:func:`relion_child_grids`). A child takes its parent's rotation and translation
    log-priors.
    """
    if not 0 < adaptive_fraction <= 1:
        raise ValueError("The adaptive fraction must be in (0, 1]")
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
    arrays = coarse.arrays
    with jax.default_device(coarse.device):
        fine_arrays = arrays._replace(
            rotations=jnp.concatenate([jnp.asarray(fine_rotations), jnp.eye(3, dtype=jnp.float32)[None]]),
            rotation_log_prior=jnp.append(jnp.repeat(arrays.rotation_log_prior[:-1], rotation_children), 0.0),
            translation_log_prior=jnp.repeat(arrays.translation_log_prior, translation_children),
            rotation_parent=jnp.asarray(np.append(rotation_parent, np.int32(n_rotations))),
            translation_parent=jnp.asarray(translation_parent),
            shift_squared=jnp.sum(jnp.asarray(fine_translations) ** 2, axis=-1),
        )
    fine = coarse._replace(
        arrays=fine_arrays,
        translations=fine_translations,
        n_coarse_rotations=n_rotations,
        n_coarse_translations=n_translations,
        block_starts=_block_starts(fine_rotations.shape[0], coarse.rotation_block_size, coarse.device),
        rotation_parent=rotation_parent,
    )
    return OversampledStream(
        coarse=coarse,
        fine=fine,
        rotation_children=int(rotation_children),
        translation_children=int(translation_children),
        adaptive_fraction=float(adaptive_fraction),
        max_significant=int(max_significant),
        job_chunk=int(job_chunk),
    )


@partial(jax.jit, static_argnames=("n_significant", "every"))
def _significant_samples(score, rows, posterior, fraction, n_real, *, n_significant, every):
    """Each image's significant coarse samples ``(B, S)``: rotation, translation and a validity mask.

    The largest posterior weights are kept until their running sum exceeds ``fraction`` of the image's mass (the
    weights are normalized), at most ``S`` of them; every kept weight is at least the last one counted, and positive.
    ``every`` (fraction 1) keeps every supported pose among the ``S`` largest. Images from ``n_real`` on (a reader's
    padding) keep none.
    """
    capacity, B, T = score.shape
    flat = jnp.transpose(score, (1, 0, 2)).reshape(B, capacity * T)
    top, position = jax.lax.top_k(flat, n_significant)
    weight = jnp.exp((top - posterior.center[:, None]) - posterior.centered_logZ[:, None])
    if every:
        valid = jnp.isfinite(top)  # every supported pose, also those whose float32 weight underflows
    else:
        crossed = jnp.cumsum(weight, axis=1) > fraction
        count = jnp.where(jnp.any(crossed, axis=1), jnp.argmax(crossed, axis=1) + 1, n_significant)
        threshold = jnp.take_along_axis(weight, (count - 1)[:, None], axis=1)
        valid = (weight >= threshold) & (weight > 0)
    valid = valid & (jnp.arange(B) < n_real)[:, None]
    return rows[position // T], (position % T).astype(jnp.int32), valid


class _FineJobs(NamedTuple):
    """One chunk of pass-2 jobs (image-major significant samples)."""

    image: jax.Array  # (J,) int32 tile image
    rotation: jax.Array  # (J,) int32 coarse rotation
    translation: jax.Array  # (J,) int32 coarse translation
    valid: jax.Array  # (J,) bool


def _job_poses(arrays, jobs, rotation_children, translation_children, n_translations_fine):
    """Child rows ``(J, R_c)``, child translations ``(J, T_c)``, tile columns ``(J, T_c)`` and log-priors."""
    rows = jobs.rotation[:, None] * rotation_children + jnp.arange(rotation_children, dtype=jnp.int32)
    shifts = jobs.translation[:, None] * translation_children + jnp.arange(translation_children, dtype=jnp.int32)
    columns = jobs.image[:, None] * n_translations_fine + shifts
    prior = arrays.rotation_log_prior[rows][:, :, None] + arrays.translation_log_prior[shifts][:, None, :]
    prior = jnp.where(jobs.valid[:, None, None], prior, -jnp.inf)
    return rows, shifts, columns, prior


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


@partial(jax.jit, static_argnames=("static", "rotation_children", "translation_children", "n_translations"))
def _score_jobs(arrays, tile, jobs, *, static, rotation_children, translation_children, n_translations):
    """Pass 2 scores of one job chunk at its children: scores ``(J, R_c, T_c)``, latent means
    ``(J, q, R_c, T_c)`` and packed covariances ``(J, R_c, tri(q))``, the dense stream's
    (:func:`_latent_block`) per job: one image, its child rotations and child translations."""
    rows, _shifts, columns, prior = _job_poses(arrays, jobs, rotation_children, translation_children, n_translations)
    planar, products = _job_projections(arrays, tile, rows, static)
    Y1 = tile.Y1[:, columns]  # (K 2F, J, T_c)
    inner = jnp.einsum("pjrk,kjt->jprt", planar, Y1, precision=_gemm(static))
    gram = jnp.einsum("mjrk,kj->jmr", products, tile.ctf2[:, jobs.image], precision=_gemm(static))

    def one(inner, gram, prior):
        score, mean, covariance = _latent_scores(inner[:, :, None], gram[:, :, None], prior[None])
        return score[:, 0], mean[:, :, 0], covariance[:, 0]

    return jax.vmap(one)(inner, gram, prior)


@partial(
    jax.jit,
    static_argnames=("static", "rotation_children", "translation_children", "n_translations", "moments"),
    donate_argnums=(0,),
)
def _moment_jobs(
    carry,
    arrays,
    tile,
    jobs,
    score,
    mean,
    covariance,
    posterior,
    *,
    static,
    rotation_children,
    translation_children,
    n_translations,
    moments,
):
    """Pass 2 posterior, M-step images and diagnostics of one job chunk, added into ``carry``.

    ``carry.rotation_mass`` is per coarse rotation: each child's posterior mass is summed into its parent.
    """
    rows, shifts, columns, _prior = _job_poses(arrays, jobs, rotation_children, translation_children, n_translations)
    J = rows.shape[0]
    centered = (score - posterior.center[jobs.image][:, None, None]) - posterior.centered_logZ[jobs.image][
        :, None, None
    ]
    gamma = jnp.exp(centered)  # (J, R_c, T_c); invalid jobs score -inf
    q = mean.shape[1]
    diagonal = [k for k, (i, j) in enumerate(zip(*np.triu_indices(q))) if i == j]
    latent_trace = sum((covariance[..., k] for k in diagonal), jnp.zeros(covariance.shape[:2], covariance.dtype))
    image_mass = jnp.sum(gamma, axis=2)  # (J, R_c)
    carry = carry._replace(
        embedding=carry.embedding.at[jobs.image].add(jnp.einsum("jrt,jqrt->jq", gamma, mean, precision=_HIGHEST)),
        rotation_mass=carry.rotation_mass.at[jobs.rotation].add(jnp.sum(image_mass, axis=1)),
        latent_covariance_trace_sum=carry.latent_covariance_trace_sum + jnp.sum(image_mass * latent_trace),
        pose_entropy_sum=carry.pose_entropy_sum - jnp.sum(jnp.where(gamma > 0, gamma * centered, 0)),
        offset_second_sum=carry.offset_second_sum + jnp.sum(gamma * arrays.shift_squared[shifts][:, None, :]),
        n_significant=carry.n_significant.at[jobs.image].add(jnp.sum(gamma > 1e-3, axis=(1, 2)).astype(jnp.int32)),
    )
    if not moments:
        return carry
    P = static.basis_size
    K = 1 if tile.frames is None else int(tile.frames.shape[0])
    F = tile.ctf2_recon.shape[1] // K
    weights = jnp.concatenate([gamma[None], gamma[None] * jnp.moveaxis(mean, 1, 0)], axis=0)  # (P, J, R_c, T_c)
    # (tri(P), J R_c, 1): the moment sums over each job's child translations.
    sums = _second_moment_sums(
        gamma.reshape(J * rotation_children, 1, -1),
        jnp.moveaxis(mean, 1, 0).reshape(q, J * rotation_children, 1, -1),
        covariance.reshape(J * rotation_children, 1, -1),
    )
    recon = tile.Y1_recon[columns]  # (J, T_c, K 2F)
    rhs_parts = jnp.einsum("pjrt,jtk->pjrk", weights, recon, precision=_gemm(static)).reshape(
        P, J * rotation_children * K, 2 * F
    )
    lhs_images = jnp.einsum(
        "mjr,jk->mjrk",
        sums.reshape(sums.shape[0], J, rotation_children),
        tile.ctf2_recon[jobs.image],
        precision=_gemm(static),
    ).reshape(sums.shape[0], J * rotation_children * K, F)
    rotations = _frame_rotations(tile, arrays.rotations[rows.reshape(-1)])
    return _scatter_moment_images(carry, arrays, static, rhs_parts, lhs_images, rotations)


def _jobs(rotation, translation, valid, chunk: int):
    """Image-major job chunks of ``chunk`` jobs from ``(B, S)`` significant samples; the last is padded."""
    B, S = valid.shape
    total = -(-(B * S) // chunk) * chunk
    pad = total - B * S

    def flat(x, fill):
        return jnp.concatenate([x.reshape(-1), jnp.full((pad,), fill, x.dtype)])

    image = flat(jnp.broadcast_to(jnp.arange(B, dtype=jnp.int32)[:, None], (B, S)), 0)
    rotation, translation, valid = flat(rotation, 0), flat(translation, 0), flat(valid, False)
    return [
        _FineJobs(image[s : s + chunk], rotation[s : s + chunk], translation[s : s + chunk], valid[s : s + chunk])
        for s in range(0, total, chunk)
    ]


@partial(jax.jit, static_argnames=("n_images",))
def _fine_posterior(score, n_images):
    """Per-image normalization of the job scores ``(J, R_c, T_c)`` (image-major, ``S`` jobs per image) and the
    first maximum's child row and translation, in (job, rotation, translation) order."""
    per_image = score.reshape(n_images, -1)
    center = jnp.max(per_image, axis=1)
    centered_logZ = jnp.log(jnp.sum(jnp.exp(per_image - center[:, None]), axis=1))
    first = jnp.argmax(per_image, axis=1)
    return center, centered_logZ, first


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


def _oversampled_tile(ostream: OversampledStream, image_indices, significant_rows, *, collect_observation, moments):
    """Both passes of one tile: the fine tile, its layout, the fine posterior, the carry and the significant samples."""
    coarse, fine = ostream.coarse, ostream.fine
    coarse_tile, _, coarse_layout = _read_tile(coarse, image_indices, significant_rows, collect_observation=False)
    kept, posterior = _score_tile(coarse, coarse_tile, coarse_layout["n_blocks"])
    n_real = _n_real(coarse_tile, coarse_layout)
    _check_finite_posterior(posterior, n_real)
    rotation, translation, valid = _significant(ostream, coarse_tile, coarse_layout, kept, posterior)
    del kept
    tile, observation_power, layout = _read_tile(
        fine, image_indices, [None] * len(image_indices), collect_observation=collect_observation
    )
    B, S = valid.shape
    jobs = _jobs(rotation, translation, valid, ostream.job_chunk)
    kwargs = dict(
        static=fine.static,
        rotation_children=ostream.rotation_children,
        translation_children=ostream.translation_children,
        n_translations=int(fine.translations.shape[0]),
    )
    scored = [_score_jobs(fine.arrays, tile, chunk, **kwargs) for chunk in jobs]
    # Padding jobs are invalid: their scores are -inf.
    score, mean, covariance = (jnp.concatenate(parts) for parts in zip(*scored))
    center, centered_logZ, first = _fine_posterior(score[: B * S], n_images=B)
    per_job = ostream.rotation_children * ostream.translation_children
    job = first // per_job
    rest = first % per_job
    top_rotation = rotation.reshape(-1)[jnp.arange(B) * S + job] * ostream.rotation_children
    fine_posterior = _Posterior(
        center=center,
        centered_logZ=centered_logZ,
        top_score=center,
        top_rotation=(top_rotation + rest // ostream.translation_children).astype(jnp.int32),
        top_translation=(
            translation.reshape(-1)[jnp.arange(B) * S + job] * ostream.translation_children
            + rest % ostream.translation_children
        ).astype(jnp.int32),
    )
    _check_finite_posterior(fine_posterior, n_real)
    fine_posterior = _drop_padding(fine_posterior, n_real)
    n_coarse = int(coarse.arrays.rotations.shape[0]) - 1
    power = jnp.float32(0) if observation_power is None else observation_power
    carry = _empty_carry(fine, B, power)._replace(rotation_mass=jnp.zeros((n_coarse,), jnp.float32))
    for index, chunk in enumerate(jobs):
        part = slice(index * ostream.job_chunk, (index + 1) * ostream.job_chunk)
        carry = _moment_jobs(
            carry,
            fine.arrays,
            tile,
            chunk,
            score[part],
            mean[part],
            covariance[part],
            fine_posterior,
            moments=moments,
            **kwargs,
        )
    return tile, layout, coarse_layout, fine_posterior, carry, (rotation, translation, valid)


@full_float32
def accumulate_oversampled_tile(ostream: OversampledStream, image_indices, significant_rows=None, *, enforce_x0=True):
    """Unregularized PPCA statistics of one tile with adaptive oversampling.

    The same statistics and diagnostics as :func:`relax.ppca_refinement.full_row_stream.accumulate_full_row_tile`,
    over each image's pass-2 poses; ``rotation_mass`` is per coarse rotation (children summed into their parent).
    The diagnostics add ``significant_samples`` (per image, the packed coarse ids ``rotation T + translation`` of
    its significant samples) and their counts.
    """
    significant_rows = [None] * len(image_indices) if significant_rows is None else significant_rows
    coarse = ostream.coarse
    with jax.default_device(coarse.device):
        tile, layout, coarse_layout, posterior, carry, (rotation, translation, valid) = _oversampled_tile(
            ostream, image_indices, significant_rows, collect_observation=True, moments=True
        )
        n_coarse = int(carry.rotation_mass.shape[0])
        layout = dict(
            layout,
            rows=np.arange(n_coarse, dtype=np.int32),
            scored_rows=coarse_layout["scored_rows"],
            supported_image_rows=coarse_layout["supported_image_rows"],
        )
        host = jax.device_get({"rotation": rotation, "translation": translation, "valid": valid})
        n_real = _n_real(tile, layout)
        packed = host["rotation"] * int(coarse.translations.shape[0]) + host["translation"]
        significant = [np.sort(packed[i][host["valid"][i]]).astype(np.int64) for i in range(n_real)]
        layout["pass2_rows"] = int(sum(s.size for s in significant)) * ostream.rotation_children
        stats = _finish_full_row_tile(coarse, image_indices, tile, layout, posterior, carry, enforce_x0=enforce_x0)
    counts = np.asarray([s.size for s in significant], np.int32)
    stats.diagnostics.update(
        {
            "engine": OVERSAMPLED_ENGINE,
            "adaptive_fraction": ostream.adaptive_fraction,
            "max_significant": ostream.max_significant,
            "significant_samples": significant,
            "significant_samples_per_image": counts,
            "significant_samples_mean": float(np.mean(counts)) if counts.size else 0.0,
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
        tile, layout, _, _, carry, _ = _oversampled_tile(
            ostream, image_indices, significant_rows, collect_observation=False, moments=False
        )
        n_real = _n_real(tile, layout)
        return DensePPCAEmbeddings(carry.embedding[:n_real], layout["original_ids"], n_real)


__all__ = [
    "OVERSAMPLED_ENGINE",
    "OversampledStream",
    "accumulate_oversampled_tile",
    "oversampled_tile_embeddings",
    "prepare_oversampled_stream",
    "relion_child_grids",
]
