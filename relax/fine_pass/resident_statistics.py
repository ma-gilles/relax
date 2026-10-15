"""Device-resident noise, norm, scale and posterior statistics for K=1 pass 2.

Stage 7 of ``../em_device_resident_pass2_design_20260918.md``. This module is
the device twin of the host bucket tail of the deleted compact engine (its inner
``_bucket_tail``): instead of pulling ~35 per-bucket arrays to the
host and accumulating them with numpy, the chunk programs of
:mod:`relax.fine_pass.resident_pass2` fold a chunk's operands into float64
accumulators that stay on the device until :func:`finalize_statistics` pulls
them once per half.

Ported production branches (the fresh K=1 guard, confirmed from
the compact engine's ``_fresh_k1_direct_noise_default`` and its bucket tail, both deleted): ``accumulate_noise=True``,
``use_relion_fine_mstep_prune=True`` (so the noise and statistics posteriors
are the same pruned ``reconstruction_probs``), RELION x-half M-step,
RELION's atomic Wavg triplet (``relion_wavg_atomic_scale_aa=True``) with its
direct low-shell noise residual, the production-algebraic per-particle norm and
no translated Wavg norm. The per-bucket operand dumps are deliberately *not*
ported; see the module's ticket report.

Layout and padding contract
---------------------------
Operands arrive in the flat, image-CSR row layout built by
:mod:`recovar.em.sparse_pass2.resident_candidates`: ``C_R`` padded candidate
rows (one per ``(image, fine rotation)`` pair) and ``C_B`` padded image slots
from the capacity ladders. Callers must guarantee that

* padded rows carry an all-zero posterior, an all-zero ``ctf_probs`` and an
  all-zero ``summed_masked`` (the kernels' own ``!= 0`` mass predicates then
  neutralise whatever is in ``proj``/``proj_abs2``), and a sentinel coarse
  rotation id;
* padded image slots carry ``image_ids = -1``, zero ``support``-bearing
  posterior rows and a zero Wavg triplet.

Scatter sentinels. ``.at[ids].add(..., mode="drop")`` does **not** drop
negative indices: JAX normalises ``-1`` to ``n - 1`` and the padded slots would
all accumulate into the last real image (measured on JAX 0.9.0.1, CPU and GPU
share the semantics). Every scatter here therefore maps padded ids to the
accumulator length, which ``mode="drop"`` does discard, and additionally zeroes
the padded values.

Numerics. Accumulators are float64. Element-wise arithmetic is copied from the
host helpers unchanged; only the association order of the reductions changes
(chunk partials instead of per-bucket host sums), which the user waived on
2026-09-18. Production precision policy is untouched: nothing here promotes the
float32 scoring, projection or Wavg streams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from relax.relion.optics_groups import pixel_rows
from relax.runtime.env_flags import parse_env_flag
from relax.runtime.shape_buckets import pow2_floor

__all__ = [
    "FinalizedClassStatistics",
    "FinalizedStatistics",
    "ResidentClassStatistics",
    "ResidentStatistics",
    "ResidentStatisticsConfig",
    "finalize_statistics",
    "make_resident_statistics",
    "resolve_statistics_config",
    "segment_sum_by_image",
]


class ResidentStatistics(NamedTuple):
    """Device float64 accumulators for one half of one pass-2 iteration.

    A ``NamedTuple`` so the whole state is one pytree that can be threaded
    through ``jax.jit`` without host round trips. The first seven fields are
    accumulated (``+=``); the per-image score/pose fields are *written* once,
    because every image belongs to exactly one chunk.

    Units follow the host accumulators they replace: ``wsum_sigma2_noise`` and
    ``wsum_img_power`` are RELION ``wsum_model`` shell sums,
    ``sigma2_offset``/``sumw`` its scalar offset and support mass,
    ``norm_correction`` the per-particle normalisation correction,
    ``scale_xa``/``scale_aa`` the per-group scale sufficient statistics and
    ``rotation_posterior_sums`` posterior mass per *coarse* rotation.
    ``best_cell`` is ``global fine rotation id * n_fine_trans + t``;
    ``best_local_rot`` is the same winner's image-local rotation slot, which is
    what the host's ``hard_assignment`` counts in.
    """

    wsum_sigma2_noise: jax.Array  # float64 [n_shells], [G, n_shells] for G optics groups
    wsum_img_power: jax.Array  # float64 [n_shells], [G, n_shells] for G optics groups
    sigma2_offset: jax.Array  # float64 []
    sumw: jax.Array  # float64 [], [G] for G optics groups
    norm_correction: jax.Array  # float64 [image_capacity]
    scale_xa: jax.Array  # float64 [n_scale_groups]
    scale_aa: jax.Array  # float64 [n_scale_groups]
    rotation_posterior_sums: jax.Array  # float64 [n_coarse_rot]
    log_evidence: jax.Array  # float64 [image_capacity]
    best_log_score: jax.Array  # float64 [image_capacity]
    max_posterior: jax.Array  # score real dtype [image_capacity]
    best_cell: jax.Array  # int64 [image_capacity], -1 until written
    score_log_z: jax.Array  # float64 [image_capacity]
    best_local_rot: jax.Array  # int32 [image_capacity], -1 until written
    invalid_best_rows: jax.Array  # int64 [], padding sanity counter
    # K>1 only: the per-class fields of RELION's class-major hidden space.
    classes: "ResidentClassStatistics | None" = None


class ResidentClassStatistics(NamedTuple):
    """Per-class accumulators of a K-class pass (RELION Class3D).

    Everything shared by the classes (noise, image power, norm and scale sums,
    the offset sum and ``sumw``) stays one total in :class:`ResidentStatistics`,
    as RELION sums it over classes (ml_optimiser.cpp:10470, :11010). The class
    axis carries only what RELION keeps per class:

    - ``log_evidence`` and ``best_log_score``: each (image, class) segment's
      log-sum-exp and maximum of the joint-min-centred scores, in the absolute
      coordinates of ``ResidentStatistics.log_evidence``;
    - ``best_cell``: the class's own winner, ``fine rotation * n_fine_trans + t``;
    - ``posterior_sums``: the pruned M-step mass of each class
      (``thr_wsum_pdf_class``, ml_optimiser.cpp:10497).

    The per-class rotation mass lives in ``ResidentStatistics.rotation_posterior_sums``
    at ``class * n_coarse_rot + coarse rotation``.
    """

    log_evidence: jax.Array  # float64 [image_capacity, K]
    best_log_score: jax.Array  # float64 [image_capacity, K]
    best_cell: jax.Array  # int64 [image_capacity, K], -1 when the class has no candidate
    posterior_sums: jax.Array  # float64 [K]


# The image axis of the per-image accumulators, and of every per-image operand
# of a half (resident_operands), is a capacity, not the half's image count:
# VDAM's subset size changes every iteration, and an image-count-shaped axis
# re-traced every program that touches it. Capacities are multiples of 256 with
# four classes per octave, so at most 25% of the axis is padding, except that a
# class is at least as wide as the smaller of its octave and 2048 rows: below
# 8192 images the padding may reach 2047 rows. VDAM's subset grows through these
# small counts over its first hundred iterations, and each new class compiled
# the resident chunk programs again (13 classes on noise1 50k, six now).
_RESIDENT_IMAGE_QUANTUM = 256
_RESIDENT_IMAGE_CLASSES_PER_OCTAVE = 4
_RESIDENT_IMAGE_MIN_CLASS_ROWS = 2048


def resident_image_capacity(n_images: int) -> int:
    """The image-axis capacity of a half with ``n_images`` images.

    A multiple of the class width, the largest of 256, a quarter of the largest
    power of two not above ``n_images`` (the octave), and the smaller of the
    octave and 2048. The classes are 256, 512, 1024, 2048, 4096, 6144, 8192,
    10240, ..., 16384, 20480, ... Rows past ``n_images`` are padding that no
    chunk addresses: chunk image slots are real image ids or negative.
    """

    if n_images <= 0:
        raise ValueError(f"n_images must be positive, got {n_images}")
    octave = pow2_floor(n_images)
    step = max(
        _RESIDENT_IMAGE_QUANTUM,
        octave // _RESIDENT_IMAGE_CLASSES_PER_OCTAVE,
        min(octave, _RESIDENT_IMAGE_MIN_CLASS_ROWS),
    )
    return -(-n_images // step) * step


@dataclass(frozen=True)
class ResidentStatisticsConfig:
    """Compile-time configuration of the statistics program.

    Hashable and passed as a ``jax.jit`` static argument, so every field here
    is part of the program's cache key. Shapes (``C_R``, ``C_B``, ``T``, ``P``,
    ``P_half``, ``P_rect``) are static through the operand avals, which is what
    keeps the number of traced programs at one per capacity class rather than
    one per bucket.
    """

    n_shells: int
    n_fine_trans: int
    # Leading extent of the per-image accumulators, resident_image_capacity of
    # the pass's image count; finalize_statistics crops it back.
    image_capacity: int
    n_coarse_rot: int
    n_scale_groups: int
    # RELION powerClass split: shells strictly above the cutoff are unweighted.
    norm_unweighted_shell_cutoff: int | None
    include_unweighted_high_shell: bool
    # Resolved on the host so the environment is part of the jit cache key.
    disable_cuda_binning: bool
    deterministic_norm_reduction: bool
    use_exact_relion_gaussian: bool
    relion_wavg_atomic_scale_aa: bool
    # ``current_size // 2 + 1`` in shell coordinates: the end of RELION's direct
    # low-shell noise residual.
    direct_noise_exclusive_shell_stop: int
    accumulate_scale: bool
    # RELION optics groups with their own noise spectrum (relax.relion.optics_groups).
    # With more than one, the noise and image-power shells are [G, n_shells] and
    # ``sumw`` is [G] (``wsum_model.sigma2_noise[g]``, ``sumw_group[g]``).
    n_optics_groups: int = 1
    # RELION Class3D classes. With more than one, ``n_coarse_rot`` counts
    # (class, coarse rotation) slots and the statistics carry a class axis
    # (:class:`ResidentClassStatistics`).
    n_classes: int = 1
    # Sum the float32 translation posterior rows into their images in row buckets
    # (segment_sum, bucket 128) instead of the scatter-add. The dense GEMM coarse
    # engine builds its config with this default; the resident passes resolve it
    # False, and the scatter-add is their image-sum order.
    float32_bucketed_image_sums: bool = True

    def __post_init__(self):
        for name in (
            "n_shells",
            "n_fine_trans",
            "image_capacity",
            "n_coarse_rot",
            "n_scale_groups",
            "direct_noise_exclusive_shell_stop",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if self.accumulate_scale and int(self.n_scale_groups) <= 0:
            raise ValueError("accumulate_scale requires at least one scale group")


def relion_wavg_atomic_triplet_runs(*, accumulate_noise: bool, scale_groups_available: bool) -> bool:
    """Whether a resident pass runs RELION's atomic Wavg triplet, and with it the direct low-shell residual.

    The triplet runs whenever the pass accumulates noise with the triplet's groups
    (``scale_groups_available``: the caller says which groups feed the triplet).
    """

    return bool(accumulate_noise and scale_groups_available)


def resolve_statistics_config(
    *,
    n_shells: int,
    n_fine_trans: int,
    n_images: int,
    n_coarse_rot: int,
    n_scale_groups: int,
    current_size: int | None,
    include_unweighted_high_shell: bool = True,
    use_exact_relion_gaussian: bool = True,
    relion_wavg_atomic_scale_aa: bool = True,
    accumulate_scale: bool = True,
    source_faithful_spectrum_norm: bool = False,
    n_optics_groups: int = 1,
    n_classes: int = 1,
    float32_bucketed_image_sums: bool = True,
) -> ResidentStatisticsConfig:
    """Build a config (``RECOVAR_DISABLE_CUDA``, the process's CUDA switch, selects the CPU shell binning).

    ``source_faithful_spectrum_norm`` mirrors
    ``sparse_pass2_wavg._weighted_image_power_shells_and_per_image``: it selects
    RELION's powerClass shell spectrum and, with it, the deterministic float64
    norm reduction.
    """

    deterministic_norm_reduction = bool(source_faithful_spectrum_norm)
    cutoff = None if current_size is None else int(current_size) // 2
    return ResidentStatisticsConfig(
        n_shells=int(n_shells),
        n_fine_trans=int(n_fine_trans),
        image_capacity=resident_image_capacity(n_images),
        n_coarse_rot=int(n_coarse_rot),
        n_scale_groups=int(n_scale_groups),
        norm_unweighted_shell_cutoff=cutoff,
        include_unweighted_high_shell=bool(include_unweighted_high_shell),
        disable_cuda_binning=parse_env_flag("RECOVAR_DISABLE_CUDA", default=False),
        deterministic_norm_reduction=bool(deterministic_norm_reduction),
        use_exact_relion_gaussian=bool(use_exact_relion_gaussian),
        relion_wavg_atomic_scale_aa=bool(relion_wavg_atomic_scale_aa),
        direct_noise_exclusive_shell_stop=int(n_shells) if cutoff is None else cutoff + 1,
        accumulate_scale=bool(accumulate_scale),
        n_optics_groups=int(n_optics_groups),
        n_classes=int(n_classes),
        float32_bucketed_image_sums=bool(float32_bucketed_image_sums),
    )


def make_resident_statistics(
    config: ResidentStatisticsConfig,
    *,
    max_posterior_dtype=jnp.float32,
) -> ResidentStatistics:
    """Zeroed accumulators matching the host arrays this stage replaces.

    ``best_cell``/``best_local_rot`` start at ``-1`` so
    :func:`finalize_statistics` can tell an image that no chunk covered from
    one whose winner is genuinely row 0, translation 0.
    """

    capacity = int(config.image_capacity)
    zeros_images = jnp.zeros(capacity, dtype=jnp.float64)
    groups = () if int(config.n_optics_groups) == 1 else (int(config.n_optics_groups),)
    return ResidentStatistics(
        wsum_sigma2_noise=jnp.zeros(groups + (int(config.n_shells),), dtype=jnp.float64),
        wsum_img_power=jnp.zeros(groups + (int(config.n_shells),), dtype=jnp.float64),
        sigma2_offset=jnp.zeros((), dtype=jnp.float64),
        sumw=jnp.zeros(groups, dtype=jnp.float64),
        norm_correction=zeros_images,
        scale_xa=jnp.zeros(int(config.n_scale_groups), dtype=jnp.float64),
        scale_aa=jnp.zeros(int(config.n_scale_groups), dtype=jnp.float64),
        rotation_posterior_sums=jnp.zeros(int(config.n_coarse_rot), dtype=jnp.float64),
        log_evidence=jnp.full(capacity, -jnp.inf, dtype=jnp.float64),
        best_log_score=jnp.full(capacity, -jnp.inf, dtype=jnp.float64),
        max_posterior=jnp.zeros(capacity, dtype=max_posterior_dtype),
        best_cell=jnp.full(capacity, -1, dtype=jnp.int64),
        score_log_z=jnp.full(capacity, -jnp.inf, dtype=jnp.float64),
        best_local_rot=jnp.full(capacity, -1, dtype=jnp.int32),
        invalid_best_rows=jnp.zeros((), dtype=jnp.int64),
        classes=None if config.n_classes == 1 else ResidentClassStatistics(
            log_evidence=jnp.full((capacity, config.n_classes), -jnp.inf, dtype=jnp.float64),
            best_log_score=jnp.full((capacity, config.n_classes), -jnp.inf, dtype=jnp.float64),
            best_cell=jnp.full((capacity, config.n_classes), -1, dtype=jnp.int64),
            posterior_sums=jnp.zeros(config.n_classes, dtype=jnp.float64),
        ),
    )


def segment_sum_by_image(
    values, row_image_local, image_capacity: int, *,
    float32_posterior_bucket_size: int | None = None,
    float32_scalar_bucket_size: int | None = None,
):
    """Sum candidate rows into their image slots.

    ``values`` has the row axis first and any trailing axes; the result has the
    image axis first. Uses the same duplicate-index scatter as the host path.
    The SPA float32 posterior can opt
    into bounded row buckets. The SPA A2/XA scalar norm terms may opt in
    separately; the default and all other callers retain the original scatter.
    """

    values = jnp.asarray(values)
    row_image_local = jnp.asarray(row_image_local, dtype=jnp.int32)
    if float32_posterior_bucket_size is not None or float32_scalar_bucket_size is not None:
        if (float32_posterior_bucket_size is not None) == (float32_scalar_bucket_size is not None):
            raise ValueError("choose exactly one float32 image reduction bucket mode")
        scalar = float32_scalar_bucket_size is not None
        bucket_size = int(float32_scalar_bucket_size if scalar else float32_posterior_bucket_size)
        if bucket_size <= 0 or values.dtype != jnp.float32 or values.ndim != (1 if scalar else 2):
            kind = "scalar rows" if scalar else "[rows, translations]"
            raise ValueError(f"bucketed image reduction needs positive bucket size and float32 {kind}")
        # `.at[ids].add` wraps valid negative indices; segment_sum drops all
        # negatives. Normalize only the original valid negative range.
        ids = jnp.where(
            (row_image_local < 0) & (row_image_local >= -image_capacity),
            row_image_local + image_capacity,
            row_image_local,
        )
        return jax.ops.segment_sum(
            values, ids, num_segments=image_capacity, bucket_size=bucket_size,
        )
    accumulator = jnp.zeros((image_capacity,) + values.shape[1:], dtype=values.dtype)
    return accumulator.at[row_image_local].add(values)


def posterior_translation_bucket_scratch_bytes(
    row_capacity: int, image_capacity: int, n_fine_trans: int, *, bucket_size: int = 128,
) -> int:
    """Conservative device reserve for a bucketed float32 ``[rows,T]`` sum.

    JAX materializes ``ceil(rows/bucket_size) × images × T`` float32 bucket
    partials. Reserve two such arrays plus the final ``[images,T]`` output
    beside the resident row stage, including its pipelined neighbour.
    """

    if min(row_capacity, image_capacity, n_fine_trans, bucket_size) <= 0:
        raise ValueError("posterior bucket capacities and size must be positive")
    partial = ((int(row_capacity) + int(bucket_size) - 1) // int(bucket_size)
               * int(image_capacity) * int(n_fine_trans) * 4)
    return 2 * partial + int(image_capacity) * int(n_fine_trans) * 4


def _drop_index(ids, length: int):
    """Map padded ids (any negative sentinel) to an index ``mode="drop"`` drops.

    ``mode="drop"`` keeps negative indices (it normalises ``-1`` to ``n - 1``),
    so a padded slot would otherwise contaminate the last real entry.
    """

    ids = jnp.asarray(ids, dtype=jnp.int32)
    return jnp.where(ids < 0, jnp.int32(length), ids)


def _flat_row_norm_and_scale_terms(
    proj,
    proj_abs2,
    summed_masked,
    ctf_probs,
    noise_variance,
    *,
    pixel_mask=None,
):
    """Per-row ``A2`` and ``XA`` partials in the flat candidate-row layout.

    Term for term this is the rectangular per-image norm residual (the test oracle
    in ``tests/oracles/projection.py``; and,
    with ``pixel_mask``, its scale-correction sibling) with the ``(B, R, P)``
    rotation axis folded into the row axis: the same masks, the same
    ``noise_variance`` broadcast and the same natural dtype promotion, summed
    over pixels only. The caller sums rows into images and forms
    ``A2 - 2 * XA`` afterwards, exactly as the host does after its
    ``axis=(1, 2)`` reduction.
    """

    ctf_has_mass = ctf_probs != 0.0
    cross_has_mass = summed_masked != 0.0
    if pixel_mask is not None:
        pixel_mask = jnp.asarray(pixel_mask, dtype=bool).reshape(-1)
        ctf_has_mass = ctf_has_mass & pixel_mask[None, :]
        cross_has_mass = cross_has_mass & pixel_mask[None, :]
    noise_variance = pixel_rows(noise_variance)
    ctf_probs_raw = jnp.where(ctf_has_mass, ctf_probs * noise_variance, 0.0)
    a2_terms = jnp.where(ctf_has_mass, proj_abs2 * ctf_probs_raw, 0.0)
    cross_terms = jnp.where(cross_has_mass, proj * jnp.conj(summed_masked), 0.0)
    xa_terms = noise_variance * cross_terms.real
    return jnp.sum(a2_terms, axis=1), jnp.sum(xa_terms, axis=1)


class FinalizedStatistics(NamedTuple):
    """Host arrays for ``make_noise_stats``/``make_relion_stats`` and the poses.

    The first seven fields are the keyword arguments of
    ``recovar.em.helpers.stats.make_noise_stats``; the next four are
    ``make_relion_stats``'s. ``hard_assignment`` counts in image-local rotation
    slots exactly like the host tail's, while ``best_fine_rotation_indices``
    are the global fine-grid ids the host reads out of
    ``per_image_inputs["oversampled_rot_indices"]``.
    """

    wsum_sigma2_noise: np.ndarray
    wsum_img_power: np.ndarray
    wsum_sigma2_offset: float
    sumw: float | np.ndarray
    wsum_norm_correction: np.ndarray
    wsum_scale_correction_xa: np.ndarray | None
    wsum_scale_correction_aa: np.ndarray | None
    log_evidence_per_image: np.ndarray
    best_log_score_per_image: np.ndarray
    max_posterior_per_image: np.ndarray
    rotation_posterior_sums: np.ndarray
    score_log_z_per_image: np.ndarray
    hard_assignment: np.ndarray
    best_fine_rotation_indices: np.ndarray
    best_translation_indices: np.ndarray
    # K>1 only; ``rotation_posterior_sums`` is then [K, n_coarse_rot].
    classes: "FinalizedClassStatistics | None" = None


class FinalizedClassStatistics(NamedTuple):
    """Host arrays of :class:`ResidentClassStatistics`, class axis first."""

    log_evidence: np.ndarray  # float64 [K, n_images]
    best_log_score: np.ndarray  # float64 [K, n_images]
    best_fine_rotation_indices: np.ndarray  # int64 [K, n_images], -1 without a candidate
    best_translation_indices: np.ndarray  # int64 [K, n_images], -1 without a candidate
    posterior_sums: np.ndarray  # float64 [K]


def finalize_statistics(
    stats: ResidentStatistics,
    *,
    config: ResidentStatisticsConfig,
    n_images: int,
    require_all_images: bool = True,
) -> FinalizedStatistics:
    """Pull the accumulators once and decode the pose fields.

    The per-image accumulators are ``config.image_capacity`` long; the rows
    past ``n_images`` are padding no chunk wrote, and are cropped here.

    Raises the padding-sanity error the host tail raises inline (a winner that
    points outside its image's candidate rows), and, unless
    ``require_all_images`` is cleared, the missing-image error that a chunk
    plan which does not cover every image would otherwise hide.
    """

    invalid = int(np.asarray(jax.device_get(stats.invalid_best_rows)))
    if invalid:
        raise RuntimeError(
            f"Resident pass-2 statistics: {invalid} image(s) selected a candidate row outside "
            "their own row range; the chunk's best-row operand points into padding"
        )
    (
        wsum_sigma2_noise,
        wsum_img_power,
        sigma2_offset,
        sumw,
        norm_correction,
        scale_xa,
        scale_aa,
        rotation_posterior_sums,
        log_evidence,
        best_log_score,
        max_posterior,
        best_cell,
        score_log_z,
        best_local_rot,
        _,
        classes,
    ) = jax.device_get(tuple(stats))
    if not 0 < n_images <= int(config.image_capacity):
        raise ValueError(f"n_images {n_images} outside the accumulators' capacity {config.image_capacity}")
    norm_correction, log_evidence, best_log_score, max_posterior, best_cell, score_log_z, best_local_rot = (
        np.asarray(values)[:n_images]
        for values in (
            norm_correction, log_evidence, best_log_score, max_posterior, best_cell, score_log_z, best_local_rot,
        )
    )

    best_cell = np.asarray(best_cell, dtype=np.int64)
    best_local_rot = np.asarray(best_local_rot, dtype=np.int64)
    if require_all_images and np.any(best_cell < 0):
        missing = int(np.count_nonzero(best_cell < 0))
        raise RuntimeError(
            f"Resident pass-2 statistics: {missing} image(s) were never written by a chunk"
        )
    best_translation_indices = np.where(best_cell < 0, -1, best_cell % config.n_fine_trans)
    best_fine_rotation_indices = np.where(best_cell < 0, -1, best_cell // config.n_fine_trans)
    hard_assignment = np.where(
        best_local_rot < 0,
        -1,
        best_local_rot * config.n_fine_trans + best_translation_indices,
    ).astype(np.int32)

    rotation_posterior_sums = np.asarray(rotation_posterior_sums, dtype=np.float64)
    finalized_classes = None
    if config.n_classes > 1:
        rotation_posterior_sums = rotation_posterior_sums.reshape(config.n_classes, -1)
        class_best_cell = np.asarray(classes.best_cell, dtype=np.int64)[:n_images].T
        finalized_classes = FinalizedClassStatistics(
            log_evidence=np.asarray(classes.log_evidence, dtype=np.float64)[:n_images].T,
            best_log_score=np.asarray(classes.best_log_score, dtype=np.float64)[:n_images].T,
            best_fine_rotation_indices=np.where(class_best_cell < 0, -1, class_best_cell // config.n_fine_trans),
            best_translation_indices=np.where(class_best_cell < 0, -1, class_best_cell % config.n_fine_trans),
            posterior_sums=np.asarray(classes.posterior_sums, dtype=np.float64),
        )

    return FinalizedStatistics(
        wsum_sigma2_noise=np.asarray(wsum_sigma2_noise, dtype=np.float64),
        wsum_img_power=np.asarray(wsum_img_power, dtype=np.float64),
        wsum_sigma2_offset=float(sigma2_offset),
        sumw=float(sumw) if np.ndim(sumw) == 0 else np.asarray(sumw, dtype=np.float64),
        wsum_norm_correction=np.asarray(norm_correction, dtype=np.float64),
        wsum_scale_correction_xa=(
            np.asarray(scale_xa, dtype=np.float64) if config.accumulate_scale else None
        ),
        wsum_scale_correction_aa=(
            np.asarray(scale_aa, dtype=np.float64) if config.accumulate_scale else None
        ),
        log_evidence_per_image=np.asarray(log_evidence, dtype=np.float64),
        best_log_score_per_image=np.asarray(best_log_score, dtype=np.float64),
        max_posterior_per_image=np.asarray(max_posterior),
        rotation_posterior_sums=rotation_posterior_sums,
        score_log_z_per_image=np.asarray(score_log_z, dtype=np.float64),
        hard_assignment=hard_assignment,
        best_fine_rotation_indices=best_fine_rotation_indices.astype(np.int64),
        best_translation_indices=best_translation_indices.astype(np.int64),
        classes=finalized_classes,
    )
