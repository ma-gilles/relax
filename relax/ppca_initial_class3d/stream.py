"""Mixture class/pose statistics using the unchanged single-class stream kernels."""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from recovar.ppca.pose_accumulators import AugmentedPPCAStats
from recovar.ppca.triangular import tri_size

from relax.helpers.half_spectrum import make_half_image_weights, make_shell_indices_half
from relax.ppca_refinement.engine import pose_invariant_score_offset
from relax.ppca_refinement.full_row_stream import (
    FullRowStream,
    _available_bytes,
    _check_finite_posterior,
    _drop_padding,
    _finish_full_row_tile,
    _MomentCarry,
    _n_real,
    _read_tile,
    _run_skipping_pass2,
    _score_tile,
    plan_tile_images,
)
from relax.ppca_refinement.full_row_stream import (
    _empty_carry as _single_empty_carry,
)
from relax.ppca_refinement.oversampled_stream import (
    OVERSAMPLED_ENGINE,
    OversampledStream,
    _chunk_sizes,
    _chunks,
    _empty_results,
    _fine_posterior,
    _job_table,
    _moment_rows,
    _pass1_tile,
    _score_jobs,
    _shift_phases,
    _significant,
    _significant_rows,
    launch_job_chunk,
)
from relax.ppca_refinement.residual_statistics import full_float32


def _empty_carry(stream, n_images, observation_power, *, moments):
    if moments:
        return _single_empty_carry(stream, n_images, observation_power)
    # Inference-only mixture passes do not allocate volume-sized moment buffers.
    return _MomentCarry(
        moments=jnp.zeros((0,), jnp.float32),
        moments_compensation=jnp.zeros((0,), jnp.float32),
        residual_power=jnp.zeros(stream.arrays.coefficient_noise.shape, jnp.float32) + observation_power,
        embedding=jnp.zeros((n_images, stream.static.basis_size - 1), jnp.float32),
        rotation_mass=jnp.zeros((len(stream.block_starts) * stream.rotation_block_size,), jnp.float32),
        latent_covariance_trace_sum=jnp.float32(0),
        pose_entropy_sum=jnp.float32(0),
        offset_second_sum=jnp.float32(0),
        n_significant=jnp.zeros((n_images,), jnp.int32),
    )


class MixtureFullRowStream(NamedTuple):
    """Class streams sharing one observation model, pose grid and tile reader."""

    streams: tuple[FullRowStream, ...]
    log_class_prior: jax.Array  # (K,), normalized; every active class has positive mass


@dataclasses.dataclass(frozen=True)
class MixtureTileResult:
    """Joint class/pose expectation for physical particles, not individual tilts.

    ``statistics[k]`` contains *joint*-posterior weighted model statistics for
    class k. Its residual numerator is only the model-dependent correction and
    its denominator is zero; use this result's shared ``residual_num/den`` for
    noise estimation. The measured image power and visible-image count occur
    once there, not K times. ``embeddings[:, k]`` is the conditional latent
    mean given class k. In inference-only mode it is accumulated with the
    conditional pose normalizer, even if the joint class mass underflows.
    During training it is zero where that class's float32 probability is zero.
    ``best_class`` selects the joint MAP class and pose, whereas a hard class
    assignment marginalized over poses is ``class_probabilities.argmax(1)``.
    Rotation and translation indices retain the supplied grid's identity.
    """

    statistics: tuple[AugmentedPPCAStats, ...]
    original_image_ids: np.ndarray
    class_probabilities: np.ndarray  # (B, K)
    embeddings: jax.Array  # (B, K, q)
    component_mass: np.ndarray  # (K,)
    best_class: np.ndarray  # (B,), joint MAP class/pose
    best_rotation_idx: np.ndarray  # (B, K), conditional MAP pose per class
    best_translation_idx: np.ndarray  # (B, K)
    max_posterior_per_image: np.ndarray  # (B,), joint class/pose Pmax
    log_likelihood: float
    residual_num: jax.Array | None
    residual_den: jax.Array | None


def prepare_mixture_full_row_stream(streams, class_prior) -> MixtureFullRowStream:
    """Validate a shared observation/grid contract once, before scoring tiles.

    Only the augmented model and angular prior may differ between classes.
    In particular the noise must be shared: a class-specific observation
    covariance would require additional Gaussian normalization constants.
    Pass-2 pruning is allowed, with the same meaning as a single-class stream;
    a zero floor retains every scored pose. Memory planning remains per class
    because only one class's pose-kept buffer is live at a time.
    """
    streams = tuple(streams)
    if not streams:
        raise ValueError("A PPCA mixture needs at least one class stream")
    first = streams[0]
    prior = np.asarray(class_prior, dtype=np.float32)
    if prior.shape != (len(streams),) or not np.all(np.isfinite(prior)) or np.any(prior <= 0):
        raise ValueError("Every PPCA class needs a finite, positive prior")
    total = prior.sum(dtype=np.float32)
    if not np.isfinite(total):
        raise ValueError("PPCA class priors must have a finite total")
    prior = prior / total
    if np.any(prior == 0):
        raise ValueError("PPCA class priors underflow after normalization")

    def same_array(a, b):
        if a is None or b is None:
            return a is b
        if isinstance(a, jax.Array) and isinstance(b, jax.Array):
            return a.shape == b.shape and bool(jax.device_get(jnp.array_equal(a, b)))
        return np.array_equal(np.asarray(a), np.asarray(b))

    array_fields = (
        "rotations", "translation_log_prior", "rotation_parent", "translation_parent",
        "score_indices", "gemm_window", "recon_indices", "coefficient_noise", "shift_squared",
    )
    scalar_fields = (
        "static", "device", "n_coarse_rotations", "n_coarse_translations", "image_batch_size",
        "rotation_block_size", "tile_images", "pass2_mass_floor",
    )
    for stream in streams[1:]:
        if stream.dataset is not first.dataset or stream.tile_loader is not first.tile_loader:
            raise ValueError("PPCA mixture classes must share the dataset and tile reader")
        if any(getattr(stream, name) != getattr(first, name) for name in scalar_fields):
            raise ValueError("PPCA mixture class geometry and execution layout must match")
        if not same_array(stream.translations, first.translations):
            raise ValueError("PPCA mixture translation grids must match")
        if any(not same_array(getattr(stream.arrays, name), getattr(first.arrays, name)) for name in array_fields):
            raise ValueError("PPCA mixture classes must share pose grids and observation noise")
        if not same_array(stream.noise_variance_half, first.noise_variance_half):
            raise ValueError("PPCA mixture classes must share observation noise")
    return MixtureFullRowStream(streams, jax.device_put(np.log(prior).astype(np.float32), first.device))


def plan_mixture_tile_images(mixture: MixtureFullRowStream, requested: int, *, retained_groups: int) -> int:
    """Conservative sequential-class plan including retained component statistics.

    The single-class planner already counts one kept pose buffer and temporary
    moment arrays. Reserve the caller's accumulated halfsets, the current K
    component results, and an additional summation copy. Models/optimizer state
    must be uploaded before this probe. This is an allocation estimate, not a
    claim of measured peak memory or performance qualification.
    """
    stream = mixture.streams[0]
    p = stream.static.basis_size
    frequencies = stream.arrays.augmented.shape[1]
    metric_channels = 1 if stream.static.metric_trace_only else tri_size(p)
    one_stat = frequencies * (4 * metric_channels + 8 * p)
    reserve = (retained_groups + 2) * len(mixture.streams) * one_stat
    available, device_bytes = _available_bytes(stream)
    if available <= reserve:
        raise MemoryError("Mixture model statistics exceed available memory; reduce K/q or image box size")
    return plan_tile_images(
        stream, requested, memory_bytes=available - reserve, device_bytes=device_bytes,
        tiles_per_call=lambda _size: 1,
    )


@jax.jit
def _mixture_normalization(centers, centered_logZ, log_class_prior):
    """Normalize class and pose jointly without summing large score offsets.

    Inputs are (B, K) conditional-pose partitions. Subtract a shared maximum
    before adding class priors, retaining the single-class score centering
    when e.g. an image's absolute score is large but class odds are small.
    ``pass2_logZ`` replaces each class's centered pose normalizer so the
    existing posterior kernels produce p(class, pose | particle).
    """
    center = jnp.max(centers, axis=1)
    relative = centers - center[:, None] + log_class_prior[None, :]
    component_logZ = relative + centered_logZ
    joint_logZ = jax.scipy.special.logsumexp(component_logZ, axis=1)
    probability = jnp.exp(component_logZ - joint_logZ[:, None])
    pass2_logZ = (center[:, None] - centers) - log_class_prior[None, :] + joint_logZ[:, None]
    best_class = jnp.argmax(relative, axis=1).astype(jnp.int32)
    pmax = jnp.exp(jnp.max(relative, axis=1) - joint_logZ)
    return center, joint_logZ, probability, pass2_logZ, best_class, pmax


@full_float32
def accumulate_mixture_full_row_tile(
    mixture: MixtureFullRowStream,
    image_indices,
    significant_rows,
    *,
    enforce_x0: bool = True,
    moments: bool = True,
) -> MixtureTileResult:
    """Shared class/pose/latent expectation, with one live class pose buffer.

    Read each observation tile once. A first sweep scores classes separately
    but retains only their conditional pose partitions and MAP indices. Their
    log partitions plus class priors give a *joint* class/pose normalizer. A
    second sweep recomputes each class's scores and conditional latent moments
    into the same buffer, then passes that joint normalizer to the existing
    posterior/backprojection kernels. Thus moments include soft class weights
    per particle; this is not independent PPCA followed by hard clustering.

    Recomputed pass-1 scores must describe exactly the same operands, grids,
    precision and model as the first sweep; otherwise its saved joint
    normalizer is invalid. No optimizer update or stochastic scoring occurs
    between the sweeps. Score classes in reverse order initially, leaving
    class zero's kept buffer available for its pass 2 without rescoring it;
    class statistics and shared-noise accumulation retain their original order.

    For cryo-ET, the stream's TOMO reader stacks all visible tilt operands of a
    physical particle before its latent Gaussian solve; the class, pose and
    latent coordinate are consequently shared across those tilts. Padding
    particles contribute no statistics. With ``moments=False`` this returns
    posteriors/embeddings only, without allocating or backprojecting volumes.
    """
    streams = mixture.streams
    first = streams[0]
    if np.asarray(image_indices).size == 0:
        raise ValueError("A mixture expectation tile must contain particles")
    with jax.default_device(first.device):
        tile, observation_power, layout = _read_tile(
            first, image_indices, significant_rows, collect_observation=moments
        )
        n_real = _n_real(tile, layout)
        conditional, kept = [None] * len(streams), None
        for k in reversed(range(len(streams))):
            kept, conditional[k] = _score_tile(streams[k], tile, layout["n_blocks"], kept)
        centers = jnp.stack([p.center for p in conditional], axis=1)
        partitions = jnp.stack([p.centered_logZ for p in conditional], axis=1)
        finite = jnp.all(jnp.isfinite(centers[:n_real])) & jnp.all(jnp.isfinite(partitions[:n_real]))
        if not bool(jax.device_get(finite)):
            raise ValueError("Every mixture particle/class needs a finite supported pose and partition")
        center, logZ, probability, pass2_logZ, best_class, pmax = _mixture_normalization(
            centers, partitions, mixture.log_class_prior
        )
        absolute_logZ = center + logZ + pose_invariant_score_offset(tile.y_norm)
        probabilities = np.asarray(jax.device_get(probability[:n_real]), dtype=np.float32)
        statistics, embeddings = [], []
        residual_power = observation_power
        for k, stream in enumerate(streams):
            if k:
                kept, _ = _score_tile(stream, tile, layout["n_blocks"], kept)
            posterior = (conditional[k]._replace(centered_logZ=pass2_logZ[:, k])
                         if moments else conditional[k])
            posterior = _drop_padding(posterior, n_real)
            # Seed measured power once, before component zero's residual
            # corrections. In K=1 this retains the legacy noise reduction
            # order, rather than adding its large baseline after pass 2.
            baseline = observation_power if moments and k == 0 else jnp.float32(0)
            carry = _empty_carry(stream, int(tile.y_norm.shape[0]), baseline, moments=moments)
            component_layout = dict(layout)
            carry = _run_skipping_pass2(
                stream, tile, kept, posterior, component_layout, carry, moments=moments
            )
            # The sufficient statistics use joint weights, whereas the saved
            # embedding has coordinates within this class's own latent space.
            if moments:
                mass = probability[:n_real, k, None]
                safe_mass = jnp.where(mass > 0, mass, jnp.float32(1))
                embeddings.append(jnp.where(mass > 0, carry.embedding[:n_real] / safe_mass, jnp.float32(0)))
            else:
                # Direct conditional accumulation avoids dividing two tiny,
                # independently rounded/underflowed joint quantities.
                embeddings.append(carry.embedding[:n_real])
            if moments:
                residual_power = carry.residual_power if k == 0 else residual_power + carry.residual_power
                correction_carry = carry._replace(residual_power=carry.residual_power - baseline) if k == 0 else carry
                stats = _finish_full_row_tile(
                    stream, image_indices, tile, component_layout, posterior, correction_carry, enforce_x0=enforce_x0
                )
                # Allocate the shared evidence by responsibility for additive
                # logging; the only likelihood for this model is the joint
                # mixture evidence returned below, not K conditional fits.
                evidence = float(jax.device_get(jnp.sum(probability[:n_real, k] * absolute_logZ[:n_real])))
                diagnostics = dict(stats.diagnostics)
                diagnostics.update(
                    engine="mixture_full_row_device_resident",
                    component_index=k,
                    component_mass=float(probabilities[:, k].sum(dtype=np.float32)),
                    log_likelihood=evidence,
                    logZ_mean=evidence / n_real,
                )
                statistics.append(dataclasses.replace(
                    stats,
                    log_likelihood=evidence,
                    residual_den=jnp.zeros_like(stats.residual_den),
                    diagnostics=diagnostics,
                ))
        if moments:
            weights = make_half_image_weights(first.static.image_shape)
            shells = make_shell_indices_half(first.static.image_shape)
            n_shells = int(np.max(np.asarray(shells))) + 1
            residual_num = jnp.zeros(n_shells, jnp.float32).at[shells].add(weights * residual_power)
            residual_den = jnp.zeros(n_shells, jnp.float32).at[shells].add(weights * layout["n_observations"])
        else:
            residual_num = residual_den = None
        best_rotation = jnp.stack([p.top_rotation[:n_real] for p in conditional], axis=1)
        best_translation = jnp.stack([p.top_translation[:n_real] for p in conditional], axis=1)
        host = jax.device_get((best_class[:n_real], best_rotation, best_translation, pmax[:n_real]))
        return MixtureTileResult(
            statistics=tuple(statistics),
            original_image_ids=np.asarray(layout["original_ids"]),
            class_probabilities=probabilities,
            embeddings=jnp.stack(embeddings, axis=1),
            component_mass=probabilities.sum(axis=0, dtype=np.float32),
            best_class=np.asarray(host[0], dtype=np.int32),
            best_rotation_idx=np.asarray(host[1], dtype=np.int32),
            best_translation_idx=np.asarray(host[2], dtype=np.int32),
            max_posterior_per_image=np.asarray(host[3], dtype=np.float32),
            log_likelihood=float(jax.device_get(jnp.sum(absolute_logZ[:n_real]))),
            residual_num=residual_num,
            residual_den=residual_den,
        )


class MixtureOversampledStream(NamedTuple):
    """Class streams of one adaptively oversampled expectation (order 1): one coarse grid, one child grid, one
    reader and one significance rule, with an augmented model per class."""

    ostreams: tuple[OversampledStream, ...]
    log_class_prior: jax.Array  # (K,), normalized


def prepare_mixture_oversampled_stream(ostreams, class_prior) -> MixtureOversampledStream:
    """Validate that the classes' oversampled streams differ only in their models and angular priors.

    The coarse and the child streams must satisfy the full-row mixture contract (grids, noise, windows, reader);
    the children counts, significance fractions, cap and job chunk must agree, as every class's jobs run on the
    same tile operands and phase factors.
    """
    ostreams = tuple(ostreams)
    if not ostreams:
        raise ValueError("A PPCA mixture needs at least one class stream")
    mixture = prepare_mixture_full_row_stream([o.coarse for o in ostreams], class_prior)
    prepare_mixture_full_row_stream([o.fine for o in ostreams], class_prior)
    first = ostreams[0]
    rule = ("rotation_children", "translation_children", "adaptive_fraction", "max_significant", "job_chunk",
            "fine_fraction")
    for o in ostreams[1:]:
        if any(getattr(o, name) != getattr(first, name) for name in rule):
            raise ValueError("PPCA mixture classes must share the oversampling rule")
        if o.base.dataset is not first.base.dataset or o.base.tile_loader is not first.base.tile_loader:
            raise ValueError("PPCA mixture classes must share the dataset and tile reader")
        if o.base.tile_images != first.base.tile_images or not bool(
            jax.device_get(jnp.array_equal(o.pass1_slots, first.pass1_slots))
        ):
            raise ValueError("PPCA mixture classes must share the pass-1 window layout")
    return MixtureOversampledStream(ostreams, mixture.log_class_prior)


@full_float32
def accumulate_mixture_oversampled_tile(
    mixture: MixtureOversampledStream,
    image_indices,
    significant_rows=None,
    *,
    enforce_x0: bool = True,
    moments: bool = True,
) -> MixtureTileResult:
    """Joint class/pose expectation of one tile with RELION's two passes per class.

    Pass 1 scores every coarse pose of each class in turn (one live pose-kept buffer) and keeps, per class, each
    particle's significant coarse samples under that class's *conditional* posterior (the same rule as the
    single-class stream, so a class's support does not shrink when its responsibility does). Pass 2 scores every
    class's jobs at their children; those results are small and are all kept, so no class is scored twice. The
    classes' fine partitions and the class priors give the joint normalizer (:func:`_mixture_normalization`);
    each class's moments are then accumulated with p(class, child pose | particle). A fine sample's joint weight is
    its conditional weight times the class's responsibility, so the class's accumulation threshold is scaled by
    that responsibility and the accumulated support is the single-class one.

    ``best_rotation_idx`` and ``best_translation_idx`` index the *child* grids (parent-major, as
    :func:`relax.ppca_refinement.oversampled_stream.relion_child_grids`); ``rotation_mass`` in each class's
    diagnostics is per coarse rotation. With ``moments=False`` the conditional posteriors weight the embeddings
    and no volume is accumulated.
    """
    ostreams = mixture.ostreams
    first = ostreams[0]
    base, fine = first.base, first.fine
    ids = np.asarray(image_indices)
    if ids.size == 0:
        raise ValueError("A mixture expectation tile must contain particles")
    significant_rows = [None] * ids.size if significant_rows is None else significant_rows
    if first.job_chunk > launch_job_chunk(first):
        raise ValueError(
            f"A chunk of {first.job_chunk} jobs exceeds one kernel launch ({launch_job_chunk(first)} jobs at most)"
        )
    with jax.default_device(first.coarse.device):
        tile, observation_power, layout = _read_tile(base, ids, [None] * ids.size, collect_observation=moments)
        n_real = _n_real(tile, layout)
        coarse_tile, coarse_layout = _pass1_tile(first, tile, layout, significant_rows)
        kept, samples = None, []
        for o in ostreams:
            kept, posterior = _score_tile(o.coarse, coarse_tile, coarse_layout["n_blocks"], kept)
            _check_finite_posterior(posterior, n_real)
            samples.append(_significant(o, coarse_tile, coarse_layout, kept, posterior))
        del kept, coarse_tile
        phases = _shift_phases(base, layout, fine.translations) if first.phases is None else first.phases[1]
        B, S = int(tile.y_norm.shape[0]), first.max_significant
        R_c, T_c = first.rotation_children, first.translation_children
        sizes = _chunk_sizes(first.job_chunk)
        q = fine.static.basis_size - 1
        tables, results, conditional, thresholds = [], [], [], []
        for o, (rotation, translation, valid, _, _) in zip(ostreams, samples):
            table = _job_table(rotation, translation, valid)
            order, _, chunks = _chunks(jax.device_get(valid), sizes, padding=B * S)
            scored = _empty_results(B * S + 1, R_c, T_c, q)
            for start, size in chunks:
                scored = _score_jobs(
                    scored, o.fine.arrays, tile, phases, table, order, np.int32(start),
                    static=o.fine.static, rotation_children=R_c, translation_children=T_c, chunk=size,
                )
            posterior, threshold = _fine_posterior(
                scored.score, rotation, translation, jnp.float32(o.fine_fraction), every=o.fine_fraction >= 1.0
            )
            _check_finite_posterior(posterior, n_real)
            tables.append(table)
            results.append(scored)
            conditional.append(posterior)
            thresholds.append(threshold)
        centers = jnp.stack([p.center for p in conditional], axis=1)
        partitions = jnp.stack([p.centered_logZ for p in conditional], axis=1)
        center, logZ, probability, pass2_logZ, best_class, pmax = _mixture_normalization(
            centers, partitions, mixture.log_class_prior
        )
        absolute_logZ = center + logZ + pose_invariant_score_offset(tile.y_norm)
        probabilities = np.asarray(jax.device_get(probability[:n_real]), dtype=np.float32)
        n_coarse = int(first.coarse.arrays.rotations.shape[0]) - 1
        statistics, embeddings = [], []
        residual_power = observation_power
        for k, o in enumerate(ostreams):
            support = _drop_padding(conditional[k], n_real)
            if moments:
                posterior = _drop_padding(conditional[k]._replace(centered_logZ=pass2_logZ[:, k]), n_real)
                threshold = thresholds[k] * probability[:, k]
            else:
                posterior, threshold = support, thresholds[k]
            baseline = observation_power if moments and k == 0 else jnp.float32(0)
            carry = _empty_carry(o.fine, B, baseline, moments=moments)._replace(
                rotation_mass=jnp.zeros((n_coarse,), jnp.float32)
            )
            row_mask = _significant_rows(results[k].score, tables[k], support, thresholds[k])
            take, n_rows, chunks = _chunks(jax.device_get(row_mask), tuple(size * R_c for size in sizes), padding=0)
            for start, size in chunks:
                carry = _moment_rows(
                    carry, o.fine.arrays, tile, phases, tables[k], results[k], posterior, threshold, take,
                    np.int32(n_rows), np.int32(start), static=o.fine.static, translation_children=T_c,
                    chunk=size, moments=moments,
                )
            if moments:
                mass = probability[:n_real, k, None]
                safe_mass = jnp.where(mass > 0, mass, jnp.float32(1))
                embeddings.append(jnp.where(mass > 0, carry.embedding[:n_real] / safe_mass, jnp.float32(0)))
                residual_power = carry.residual_power if k == 0 else residual_power + carry.residual_power
                correction_carry = carry._replace(residual_power=carry.residual_power - baseline) if k == 0 else carry
                component_layout = dict(
                    layout, rows=np.arange(n_coarse, dtype=np.int32), scored_rows=coarse_layout["scored_rows"],
                    supported_image_rows=coarse_layout["supported_image_rows"], pass2_rows=n_rows,
                )
                stats = _finish_full_row_tile(
                    o.coarse, ids, tile, component_layout, posterior, correction_carry, enforce_x0=enforce_x0
                )
                evidence = float(jax.device_get(jnp.sum(probability[:n_real, k] * absolute_logZ[:n_real])))
                _, _, valid, sample_mass, capped = jax.device_get(samples[k])
                diagnostics = dict(stats.diagnostics)
                diagnostics.update(
                    engine="mixture_" + OVERSAMPLED_ENGINE,
                    component_index=k,
                    component_mass=float(probabilities[:, k].sum(dtype=np.float32)),
                    log_likelihood=evidence,
                    logZ_mean=evidence / n_real,
                    adaptive_fraction=o.adaptive_fraction,
                    max_significant=o.max_significant,
                    fine_fraction=o.fine_fraction,
                    scored_fine_rows=int(valid[:n_real].sum()) * R_c,
                    significant_samples_per_image=np.asarray(valid[:n_real].sum(axis=1), np.int32),
                    significant_mass_per_image=np.asarray(sample_mass[:n_real], np.float32),
                    significant_capped_per_image=np.asarray(capped[:n_real], bool),
                )
                statistics.append(dataclasses.replace(
                    stats, log_likelihood=evidence, residual_den=jnp.zeros_like(stats.residual_den),
                    diagnostics=diagnostics,
                ))
            else:
                embeddings.append(carry.embedding[:n_real])
        if moments:
            weights = make_half_image_weights(first.coarse.static.image_shape)
            shells = make_shell_indices_half(first.coarse.static.image_shape)
            n_shells = int(np.max(np.asarray(shells))) + 1
            residual_num = jnp.zeros(n_shells, jnp.float32).at[shells].add(weights * residual_power)
            residual_den = jnp.zeros(n_shells, jnp.float32).at[shells].add(weights * layout["n_observations"])
        else:
            residual_num = residual_den = None
        best_rotation = jnp.stack([p.top_rotation[:n_real] for p in conditional], axis=1)
        best_translation = jnp.stack([p.top_translation[:n_real] for p in conditional], axis=1)
        host = jax.device_get((best_class[:n_real], best_rotation, best_translation, pmax[:n_real]))
        return MixtureTileResult(
            statistics=tuple(statistics),
            original_image_ids=np.asarray(layout["original_ids"]),
            class_probabilities=probabilities,
            embeddings=jnp.stack(embeddings, axis=1),
            component_mass=probabilities.sum(axis=0, dtype=np.float32),
            best_class=np.asarray(host[0], dtype=np.int32),
            best_rotation_idx=np.asarray(host[1], dtype=np.int32),
            best_translation_idx=np.asarray(host[2], dtype=np.int32),
            max_posterior_per_image=np.asarray(host[3], dtype=np.float32),
            log_likelihood=float(jax.device_get(jnp.sum(absolute_logZ[:n_real]))),
            residual_num=residual_num,
            residual_den=residual_den,
        )
