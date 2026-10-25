"""Device-resident K=1 sparse pass-2 driver (T9b integration).

This module wires the five device-resident pass-2 stage modules into one
driver with the signature and return type of the compact engine's
``compute_pass2_stats_sparse_bucketed`` (deleted):

* T5 :mod:`relax.fine_pass.resident_candidates` -- the flat image-CSR
  candidate table and its fixed-capacity chunks;
* T6 :mod:`relax.fine_pass.resident_scoring` -- the resident per-image
  scoring operands and the one-program-per-capacity-class scoring stage;
* T7 ``relax.cuda.posterior.sparse_pass2_segmented_*`` -- the segmented
  RELION float32 fine posterior over those CSR segments;
* T8 ``relax.cuda.fine.relion_wavg_*flat_rows*`` -- the flat-row RELION
  Wavg triplet and its atomic accumulation;
* T9a :mod:`relax.fine_pass.resident_statistics` -- the device float64
  accumulators, their finalization and the pose decode.

Selection and scope
-------------------
The driver is relax's one pass-2 engine (``relax.fine_pass.dispatch``). It runs the
production configuration (RELION x-half M-step, exact RELION fine Gaussian scoring, float32
fine posterior, fine M-step prune, float32 scoring, no diagnostics or dumps); every other
configuration raises :class:`ResidentConfigurationUnsupported` naming the missing piece.
There is no other engine to fall back to. It is checked against the NumPy RELION E-step
reference (``tests/unit/test_resident_relion_reference.py``).

What is deliberately different from the compact engine
------------------------------------------------------
Three differences are layout or reduction-order changes, not arithmetic
changes, and each is measured rather than assumed:

1. **Translation application.** The compact K=1 route scores a pre-shifted
   ``(B, T, N)`` image tile; this driver calls the flat-row *fused translate*
   kernel, which applies the translation phase per pixel inside the scoring
   kernel. T6 measured the two to agree bitwise on every score-window pixel and
   to differ only on the ``ky = -N/2`` Nyquist row of a full (unwindowed) half
   image. The production window excludes that row, and the configuration gate
   below refuses the unwindowed case.
2. **log-Z reduction order.** The compact route reduces ``sum exp`` with an XLA
   float64 tree over ``(B, R*T)``; the segmented CUDA handler reduces per
   segment in block order. The posteriors themselves come from the float32
   ``sum_weight`` scan, which T7 showed is bitwise identical to the rectangular
   handler, so only ``log_evidence``/``score_log_z`` can move.
3. **Statistics reduction order.** Chunk partials replace per-bucket host sums.
   The user waived reduction-order parity for these accumulators on 2026-09-18;
   the same-source band is the gate.

Pixel-axis blocking
-------------------
Stages 5-7 carry a pixel axis (``N_recon``, 4324 at the hp3 production state),
so a whole chunk's ``proj``/``summed``/``ctf_probs`` would be tens of gigabytes
at the largest capacity class. The design anticipates this ("project in row
blocks inside the chunk program"). The driver therefore walks each chunk in
fixed-size row blocks, so the pixel-axis programs are keyed on one static block
shape regardless of the chunk's capacity class. Because of that split, the
image-level statistics are folded once per chunk by
:func:`~relax.fine_pass.resident_image_terms.accumulate_chunk_image_terms` (T9a's arithmetic, with the row-pixel
reductions arriving as chunk partials) rather than by T9a's single fused
program, which assumes one call per chunk with the whole pixel axis resident.
"""

from __future__ import annotations

import logging
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from recovar.reconstruction import noise as noise_utils

import relax.runtime.engine_tuning
from relax.fine_pass import (
    projection_blocks,
    resident_block_programs,
    resident_chunk_inputs,
    resident_chunk_memory,
    resident_chunk_programs,
    resident_chunk_records,
    resident_chunk_run,
    resident_configuration,
    resident_image_terms,
    resident_mstep_blocks,
    resident_pass_tables,
    resident_projections,
    resident_stable_windows,
    resident_statistics,
    resident_table_blocks,
)
from relax.fine_pass.bucket_io import relion_cuda_score_translation_angles_if_available
from relax.fine_pass.policy import (
    ResidentConfigurationUnsupported,
    projection_cache_enabled_for_pass,
)
from relax.fine_pass.projection_blocks import (
    compute_sparse_pass2_windowed_projections_block,
    project_rows_by_class,
    projection_kwargs_for_relion_score_window,
)
from relax.fine_pass.resident_candidates import (
    plan_pass_chunks,
)
from relax.fine_pass.resident_operands import (
    ResidentOperandsUnsupported,
    describe_resident_operand_mismatch,
    prepare_resident_half_operands,
    resident_half_operand_avals,
    resident_half_operand_presence,
    resident_operands_max_bytes,
    resident_prepare_batch_size,
    resident_prepare_reserved_bytes,
)
from relax.fine_pass.resident_significance import (
    fine_rotation_children,
)
from relax.fine_pass.scoring import (
    relion_native_fine_units_enabled,
    relion_powerclass_noise_dtypes,
)
from relax.fine_pass.window import (
    pass2_half_weights,
    pass2_window_setup,
    sparse_pass2_window_setup,
)
from relax.fourier.preprocessing import half_translation_phase_table
from relax.projection.projection import relion_kernel_zero_rows
from relax.reconstruction.half_volume_mstep import (
    allocate_row_partial,
    half_volume_accumulator_shape,
    relion_x_half_mstep_accumulator_dtypes,
)
from relax.relion import ctf
from relax.relion.optics_aberrations import dataset_magnification_is_anisotropic, dataset_needs_exact_ctf
from relax.relion.scale_groups import prepare_scale_correction_groups
from relax.runtime.compile_ahead import CompileAheadPool
from relax.runtime.memory_budget import (
    current_device_available_bytes,
    device_available_bytes,
    device_free_memory_bytes,
    jax_allocator_free_memory_bytes,
    jax_allocator_pool_free_bytes,
    max_adjoint_block_bytes_for_pass,
    max_translation_tile_bytes_for_pass,
    projection_cache_build_max_rotations_per_call,
    projection_cache_fits_budget,
    projection_cache_max_bytes_for_pass,
    projection_cache_transient_bytes,
)
from relax.sampling.translation_prior import validate_translation_prior_centers
from relax.types import SparsePass2Output, make_relion_stats

logger = logging.getLogger(__name__)

# The resident pass sums its float32 posterior and norm rows into their images with
# the scatter-add; the bucketed segment_sum (and its planned scratch) is the dense
# GEMM coarse engine's.
_FLOAT32_BUCKETED_IMAGE_SUMS = False

# Row capacities are multiples of the M-step block so every chunk decomposes
# into whole blocks; image capacities follow the design's ladder. The design's
# three classes, restored after the denser five-class ladder was measured and
# bought nothing: at hp3 the matched pairs put the two ladders inside the
# control's own drift (loop 22.6 versus 22.1 s per half) and at the early state
# they are indistinguishable (resident warm 67.6 / 60.4 versus 67.0 / 61.5 s,
# occupancy 0.95-0.97 either way), while the dense
# ladder costs 84 extra traced programs. Every extra class is one more program
# per capacity-class stage.
# The occupancy of each plan is still logged, so a future change has its number.
# 524288 is for early low-SNR states, where each image keeps thousands of rows:
# Class3D K4 100k/256 snr05 iteration 2 ran 8722 chunks of 131072 rows, 2028 of
# 524288, chunk loop 619 -> 523 s and iteration 3 453 -> 374 s.
# The cached-path gather budget (cached_row_capacity_ladder) and the chunk
# memory plan drop it wherever a chunk that size does not fit.
_DEFAULT_ROW_CAPACITY_LADDER = (8192, 32768, 131072, 524288)
_DEFAULT_IMAGE_CAPACITY_LADDER = (32, 128, 512)

__all__ = [
    "ResidentClassInputs",
    "ResidentKClassPass2Output",
    "compute_k_class_pass2_stats_resident",
    "compute_pass2_stats_resident",
    "ResidentPass2Result",
]


def _resident_wavg_arithmetic(
    *,
    source_faithful_spectrum_norm,
):
    """Resolve the pass's Wavg arithmetic: RELION's powerClass spectrum and its exact BPref operands.

    Returns ``(spectrum_norm, exact_bpref_operands)``; whether the atomic triplet runs is
    ``resident_pass_tables.resident_scale_and_triplet``'s. The resident statistics stage implements
    RELION's atomic Wavg triplet only, for fresh passes and subset or focused replays alike (user
    decision 2026-09-26: the replays' earlier unordered, non-atomic arithmetic was the compact engine's).
    """

    # A pass that preserves RELION's BPref order (every fresh K=1 pass) runs RELION's
    # powerClass spectrum with its exact BPref operands; there is no other variant.
    spectrum_norm = exact_bpref_operands = bool(source_faithful_spectrum_norm)
    return spectrum_norm, exact_bpref_operands


def _require_gpu_pass2() -> None:
    """Fail clearly without a GPU: the resident driver is relax's only pass 2 (no CPU back end).

    Every resident stage is a CUDA FFI target (user decision 2026-09-26: pass 2 is GPU-only;
    CPU correctness tests use the NumPy RELION E-step reference instead).
    """

    from recovar import cuda_backproject

    backend = jax.default_backend()
    if backend != "gpu" or not cuda_backproject.custom_cuda_requested():
        raise RuntimeError(
            "relax's pass 2 runs only on the device-resident engine, which needs a CUDA GPU "
            f"and relax's custom CUDA library (JAX backend {backend!r}, custom CUDA "
            f"{'available' if cuda_backproject.custom_cuda_requested() else 'unavailable'}). "
            "There is no CPU pass 2: run on a GPU node."
        )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _coarse_normalization_reuse(
    tables,
    *,
    relion_f32_normalization_sum_weight,
    relion_coarse_hard_assignment,
    relion_coarse_max_posterior,
    fine_rotation_parent,
    fine_translation_parent,
    max_posterior_dtype,
    relion_coarse_class_assignment=None,
) -> resident_chunk_records.CoarseNormalizationReuse | None:
    """The retained coarse normalization of a zero-oversampling pass, on the device.

    ``None`` unless the caller supplied the coarse float32 sum; the production
    gate requires the winner and Pmax with it.
    """

    from relax.fine_pass.resident_candidates import coarse_winner_cells

    if relion_f32_normalization_sum_weight is None:
        return None
    sum_weight = np.asarray(relion_f32_normalization_sum_weight, dtype=np.float64).reshape(-1)
    if sum_weight.shape != (tables.n_images,):
        raise ValueError(
            f"relion_f32_normalization_sum_weight must have shape ({tables.n_images},), got {sum_weight.shape}"
        )
    max_posterior = np.asarray(relion_coarse_max_posterior, dtype=np.float64).reshape(-1)
    if (
        max_posterior.shape != (tables.n_images,)
        or not np.all(np.isfinite(max_posterior))
        or np.any(max_posterior < 0)
        or np.any(max_posterior > 1)
    ):
        raise ValueError("coarse Pmax must be one finite value in [0, 1] per image")
    winner_cell = coarse_winner_cells(
        tables,
        relion_coarse_hard_assignment,
        fine_rotation_parent=fine_rotation_parent,
        fine_translation_parent=fine_translation_parent,
        coarse_classes=relion_coarse_class_assignment,
    )
    # Stored at the image capacity like every per-image device array, so the
    # chunk programs that read them are not keyed on the subset size. Chunks
    # only read real image ids.
    padding = resident_statistics.resident_image_capacity(tables.n_images) - tables.n_images
    return resident_chunk_records.CoarseNormalizationReuse(
        sum_weight=jnp.asarray(np.pad(sum_weight, (0, padding)), dtype=jnp.float64),
        max_posterior=jnp.asarray(np.pad(max_posterior, (0, padding)), dtype=max_posterior_dtype),
        winner_cell=jnp.asarray(np.pad(np.asarray(winner_cell), (0, padding), constant_values=-1), dtype=jnp.int64),
    )


class ResidentClassInputs(NamedTuple):
    """The class axis of a K-class pass (RELION Class3D), one entry per class.

    ``rotation_log_priors`` are each class's coarse rotation log prior with its
    ``log pdf_class`` folded in, as the compact engine takes them
    (k_class.py ``_class_rotation_prior``).
    """

    significant_sample_indices: tuple
    rotation_log_priors: tuple


def _rows_within_rotated_sphere(label, r_max, rotations):
    """``(sigma_min * label) ** 2 < r_max ** 2 + 1``, ``sigma_min`` the smallest singular value of all ``rotations``.

    The pass's rows are one projection matrix ``P`` composed with rotations, so every ``A^T A`` (or ``A A^T``) is
    ``P^T P`` (``P P^T``) up to rounding. By Weyl's inequality the smallest eigenvalue over all rows is at least the
    first row's minus the largest Frobenius distance to it, and at most the first row's: ``sigma_min`` lies in that
    bracket. The predicate falls with ``sigma_min``, so when both ends give the same rows they are the rows of the
    exact value; otherwise (rows that do not share one ``P``) every matrix's SVD decides, as it always did. The
    bracket replaces an SVD of every fine rotation of every pass (0.33 s a pass on multioptics VDAM).
    """

    a = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
    bound = float(r_max) ** 2 + 1.0

    def rows_at(sigma):
        return (sigma * label) ** 2 < bound

    first = float(np.linalg.svd(a[0], compute_uv=False).min())
    low = 0.0
    for gram in (np.einsum("nki,nkj->nij", a, a), np.einsum("nik,njk->nij", a, a)):
        spread = float(np.sqrt(np.max(np.sum((gram - gram[0]) ** 2, axis=(1, 2)))))
        low = max(low, float(np.sqrt(max(0.0, np.linalg.eigvalsh(gram[0]).min() - spread))))
    low = min(low, first)
    if np.array_equal(rows_at(low), rows_at(first)):
        return rows_at(first)
    return rows_at(float(np.linalg.svd(a, compute_uv=False).min()))


def _padded_sgd_residual_rows(rows):
    """``_sgd_residual_rows``' ``(positions, pixels)`` padded to a power-of-two count by repeating the last pixel.

    The count follows the logical current size and the model radius, so unpadded it gave the SGD projection, the
    cache writes and the M-step block programs a new shape on most passes (about 390 compiles, 30 s on multioptics
    VDAM). A repeated pixel projects the same value into the same reconstruction position, which the M-step
    ``set`` writes again, so the residual is unchanged.
    """

    if rows is None:
        return None
    positions, pixels = rows
    capacity = 1 << max(0, int(positions.size) - 1).bit_length()
    pad = capacity - int(positions.size)
    return np.concatenate([positions, np.repeat(positions[-1:], pad)]), np.concatenate([pixels, np.repeat(pixels[-1:], pad)])


def _padded_window_union(union):
    """A score-only :class:`ProjectionWindowUnion` with its pixels padded to a power-of-two count (the last repeated).

    The SGD rows' distinct pixels change in number with the current size and radius like the rows themselves
    (:func:`_padded_sgd_residual_rows`); the repeated pixel is projected and never taken. A window pixel outside
    the crop takes the zero column, which moves to the padded end, as in
    :func:`relax.fine_pass.projection_blocks.capacity_projection_window_union`.
    """

    indices = np.asarray(union.indices, dtype=np.int64)
    if indices.size == 0 or union.recon_take is not None:
        return union
    capacity = 1 << (int(indices.size) - 1).bit_length()
    take = np.asarray(union.score_take)
    return union._replace(
        indices=jnp.asarray(np.pad(indices, (0, capacity - indices.size), mode="edge").astype(np.int32)),
        score_take=jnp.asarray(np.where(take == indices.size, capacity, take).astype(np.int32)),
    )


def _sgd_residual_rows(recon_window_indices, *, image_shape, projector_output_size, r_max, rotations):
    """The reconstruction pixels of VDAM's subtracted reference that the projection cache zeroes, or None.

    RELION's fine diff2 and wavg kernels zero the rows past the model radius of an image
    window wider than the model sphere (:func:`relax.projection.projection.relion_kernel_zero_rows`),
    and the cache holds those projections. The SGD backprojection kernel projects every row
    of the reference it subtracts (BP.cuh:476-492): for an optics group on a coarser grid
    (scale s > 1) those rows hold pixels inside the rotated sphere. Returns
    ``(positions, pixels)``, their positions in the reconstruction window and their centred
    half-image indices, or None when the window has none (every single-grid pass).

    Only rows that can reach the sphere count: a pixel of row label ``y`` lies at rotated radius
    at least ``sigma_min * |y|`` (``sigma_min`` the smallest singular value of the pass's
    projection ``rotations``), and the kernel projects it only when ``int(radius^2) <= r_max^2``.
    A window one row wider at ``s = 1 + 1e-8`` (a real-data header pixel) then has none.
    """

    if projector_output_size is None or r_max is None or int(r_max) <= 0:
        return None
    zero = relion_kernel_zero_rows(int(image_shape[0]), int(projector_output_size), int(r_max), "fine")
    if zero is None:
        return None
    n = int(image_shape[0])
    label = np.arange(n) - n // 2
    if int(projector_output_size) == n:
        label[0] = n // 2  # the full window's positive Nyquist row (relion_kernel_zero_rows)
    reachable = _rows_within_rotated_sphere(label, r_max, rotations)
    zero = np.asarray(zero) & np.repeat(reachable, n // 2 + 1)
    recon = np.asarray(recon_window_indices, dtype=np.int64)
    positions = np.flatnonzero(zero[recon])
    if positions.size == 0:
        return None
    return positions.astype(np.int32), recon[positions].astype(np.int32)


def _resident_pass2(
    experiment_dataset,
    volume,
    noise_variance,
    translations,
    significant_sample_indices,
    nside_level,
    disc_type,
    *,
    engine_tuning: relax.runtime.engine_tuning.EngineTuning,
    scoring_overrides,
    oversampling_order,
    current_size,
    reconstruction_current_size=None,
    wsum_current_size=None,
    translation_step,
    rotation_log_prior,
    score_with_masked_images,
    return_stats,
    translation_log_prior,
    accumulate_noise,
    half_spectrum_scoring,
    projection_padding_factor,
    projection_mask_current_image_disk=False,
    reconstruction_padding_factor,
    image_corrections,
    scale_corrections,
    image_pre_shifts,
    use_float64_scoring,
    translation_prior_centers=None,
    do_gridding_correction=False,
    square_window=False,
    random_perturbation,
    group_ids=None,
    scale_correction_group_count=None,
    scale_correction_data_vs_prior=None,
    normalization_log_z=None,
    relion_f32_normalization_sum_weight=None,
    relion_coarse_hard_assignment=None,
    relion_coarse_max_posterior=None,
    relion_coarse_class_assignment=None,
    normalization_other_score_log_z=None,
    return_score_log_z=False,
    return_score_log_z_only=False,
    fine_source_eulers_override=None,
    return_source_eulers=False,
    fine_rotations_override=None,
    fine_mstep_rotations_override=None,
    fine_rotation_parent_override=None,
    fine_translations_override=None,
    fine_translation_parent_override=None,
    relion_x_half_mstep=False,
    mstep_subtract_ctf_projection=False,
    relion_fine_mstep_prune=False,
    relion_firstiter_score_mode="gaussian",
    relion_firstiter_winner_take_all=False,
    relion_exact_fine_gaussian=True,
    relion_f32_fine_posterior=False,
    relion_exact_fine_normalized_cc=False,
    relion_projector_half=None,
    relion_projector_texture=None,
    relion_projector_r_max=None,
    adaptive_fraction=0.999,
    include_unweighted_norm_high_shell: bool = True,
    source_faithful_spectrum_norm: bool = False,
    symmetry_label: str = "C1",
    keep_physical_bpref: bool = False,
    relion_translation_angle_scale: float = 1.0,
    optics_group_ids=None,
    reconstruction_volume_current_size=None,
    reconstruction_image_radius=None,
    reconstruction_group_ids=None,
    reconstruction_group_count=None,
    classes: ResidentClassInputs | None = None,
    tilt=None,
    coarse_rotation_ids=None,
    unit_rotation_log_prior=None,
    full_grid_engine=None,
    nyquist_column_counting: str = "relion",
    firstiter_cc_support: str = "relion",
    image_translations=None,
):
    """The device-resident sparse pass 2 over one or K classes; returns ``ResidentPass2Result``.

    With ``classes`` set, ``volume`` (and ``relion_projector_half``, when given)
    stack the K class references on the leading axis, and
    ``significant_sample_indices``/``rotation_log_prior`` are unused (None): each
    class's support and prior come from ``classes``. Every image's posterior
    segment then spans all classes (docs/development/resident_segments.md).
    :func:`compute_pass2_stats_resident` is the K=1 entry and
    :func:`compute_k_class_pass2_stats_resident` the K-class one.

    See the module docstring for what is layout-equal to the compact engine and
    what is a deliberate reduction-order change. The configuration gate runs
    before any device work, so an unsupported pass fails immediately instead of
    part way through a half.

    ``noise_variance`` is one shared spectrum, or ``[G, P]`` rows of G optics
    groups with ``optics_group_ids`` giving each image's row; each image then
    scores, backprojects and adds its noise sums with its own group
    (:mod:`relax.relion.optics_groups`), and the noise statistics come back per group.

    ``reconstruction_volume_current_size`` is the backprojector's model size when the
    images are on another grid than the reference (an optics group with another box
    or pixel size): the image-side windows keep ``reconstruction_current_size`` in image
    pixels, the accumulator and its adjoint radius use the reference model size.

    ``nyquist_column_counting`` and ``firstiter_cc_support`` are the consistency options of the
    per-image sums (docs/math/relion_consistency_options.md). They change only arrays built
    here: the scoring weights (``pass2_half_weights``), the image the noise statistics read
    and the normalized-CC image power (``prepare_unshifted_bucket_operands``). Subtomogram and
    full-grid GEMM passes refuse them.

    ``image_translations`` ``[n_images, 2]`` gives each image its own translation sample (RELION
    ``--skip_align``): the images are translated as they are prepared
    (``prepare_unshifted_bucket_operands``) and the fine grid is the one zero translation.

    ``full_grid_engine`` is the caller's full-grid engine (``relax.dense.gemm_coarse_engine.run_dense_gemm_full_grid``)
    or None: with one, the pass resolves its operands, hands them to it as a
    :class:`relax.fine_pass.full_grid.DenseGemmPreparedState` instead of building candidate tables, and returns
    its result.

    ``wsum_current_size`` is the image current size of the weighted sums (RELION's
    ``image_current_size`` in ``storeWeightedSums``, ml_optimiser.cpp:6803-6806) when the E-step
    scores below it (``--strict_highres_exp``): the noise shells, the Wavg rectangle and its
    scale mask, the powerClass terms (``highres_Xi2`` and the norm's high shell, both summed
    above ``image_current_size``, ml_optimiser.cpp:6381-6404) and the norm cutoff use it, while
    ``current_size`` stays the scoring window. None (the default) is ``current_size``.
    """

    if (nyquist_column_counting != "relion" or firstiter_cc_support != "relion") and (
        tilt is not None or full_grid_engine is not None
    ):
        raise NotImplementedError(
            "nyquist_column_counting / firstiter_cc_support are implemented for the single-particle "
            "resident pass 2 only (not subtomogram or full-grid GEMM passes)"
        )

    from relax.fine_pass.window import (
        fine_translation_log_prior_2d,
        pass2_projection_budget,
        pass2_relion_flags,
    )
    from relax.sampling.grids import rotation_grid_size
    from relax.sampling.oversampled_grids import get_oversampled_translation_grid, infer_translation_step
    from relax.sampling.symmetry import canonicalize_rotational_symmetry

    _require_gpu_pass2()
    overall_t0 = time.time()
    (
        use_exact_relion_gaussian,
        use_relion_f32_fine_posterior,
    ) = pass2_relion_flags(
        relion_exact_fine_gaussian=relion_exact_fine_gaussian,
        relion_firstiter_score_mode=relion_firstiter_score_mode,
        relion_f32_fine_posterior=relion_f32_fine_posterior,
        f32_fine_posterior_override=scoring_overrides.f32_fine_posterior,
    )
    firstiter_cc = relion_firstiter_score_mode == "normalized_cc"
    if classes is None:
        class_supports = (significant_sample_indices,)
        class_rotation_priors = (rotation_log_prior,)
    else:
        if significant_sample_indices is not None or rotation_log_prior is not None:
            raise ValueError(
                "a K-class pass takes each class's support and rotation prior from `classes`"
            )
        class_supports = tuple(classes.significant_sample_indices)
        class_rotation_priors = tuple(classes.rotation_log_priors)
        if len(class_supports) != len(class_rotation_priors) or len(class_supports) < 2:
            raise ValueError("a K-class pass needs K >= 2 supports and rotation priors")
    n_classes = len(class_supports)
    if n_classes > 1:
        # RELION's Class3D E-step; the --firstiter_cc winner is a K=1 pass per class.
        if full_grid_engine is None:
            resident_configuration.require(not firstiter_cc, "the K-class resident pass scores the Gaussian likelihood")
        # Zero oversampling (relax#73): the coarse sum, Pmax and winner come with the winner's class.
        resident_configuration.require(
            (relion_f32_normalization_sum_weight is None) == (relion_coarse_class_assignment is None),
            "the K-class zero-oversampling pass reuses the coarse normalization with the coarse winner's class",
        )
        resident_configuration.require(
            relion_f32_normalization_sum_weight is None
            or (
                relion_coarse_hard_assignment is not None
                and relion_coarse_max_posterior is not None
                and int(oversampling_order) == 0
            ),
            "the coarse normalization sum is reused with the coarse winner and Pmax, at zero oversampling only",
        )
    else:
        resident_configuration.require(relion_coarse_class_assignment is None, "a coarse winner class belongs to a K-class pass")

    n_images = int(experiment_dataset.n_units)
    # Subtomogram particles (S4.2, resident_tilts): the posterior unit is the particle; the dataset's
    # rows are its tilt images. The candidate tables and the statistics are per particle.
    n_units = n_images if tilt is None else int(np.asarray(tilt.unit_image_offsets).size - 1)
    if tilt is not None:
        resident_configuration.require(
            relion_f32_normalization_sum_weight is None,
            "subtomogram particles run the fine pass without zero-oversampling reuse",
        )
        resident_configuration.require(
            int(np.asarray(tilt.unit_image_offsets)[-1]) == n_images,
            "the tilt layout must cover every image of the half",
        )
        # The offset prior is the particle's (tilt.unit_translation_prior), not an image's.
        resident_configuration.require(
            translation_log_prior is None and translation_prior_centers is None,
            "a tilt pass takes its offset prior from tilt.unit_translation_prior",
        )
    n_coarse_trans = int(np.asarray(translations).shape[0])
    symmetry_label = canonicalize_rotational_symmetry(symmetry_label)
    # The coarse grid is RELION's asymmetric-unit HEALPix sampling
    # (healpix_sampling.cpp removeSymmetryEquivalentPoints); the caller's fine
    # rotation override already holds its children.
    n_coarse_rot = rotation_grid_size(nside_level, symmetry_label)
    if coarse_rotation_ids is not None or unit_rotation_log_prior is not None:
        # A subtomogram local search (relax.steps.tomo_half): its coarse grid is the union of the
        # particles' local rotations (coarse id c is grid rotation coarse_rotation_ids[c]) and each
        # particle carries its own orientation prior over its support's coarse rotations (K classes: over each
        # class's support, ``unit_rotation_log_prior`` then holding one list per class).
        resident_configuration.require(tilt is not None, "a compact coarse grid is the subtomogram local search's")
        resident_configuration.require(
            rotation_log_prior is None and all(prior is None for prior in class_rotation_priors),
            "a local search's rotation priors are the particles' own",
        )
        if coarse_rotation_ids is not None:
            n_coarse_rot = int(np.asarray(coarse_rotation_ids).size)
    image_shape = experiment_dataset.image_shape
    # Some images CTF-premultiplied: the resident operands carry one more array.
    ctf_premultiplied_pass = ctf.dataset_has_premultiplied_ctf(experiment_dataset, image_shape)
    if ctf_premultiplied_pass and nyquist_column_counting != "relion":
        # RELION's average CTF^2 of premultiplied images divides by Npix_per_shell, which keeps RELION's count.
        raise NotImplementedError("nyquist_column_counting is not implemented for CTF-premultiplied images")
    volume_shape = experiment_dataset.volume_shape
    # Anisotropic magnification: the M-step clips on RELION's rotated radius (adjoint.ReferenceSphereClip).
    anisotropic_magnification = dataset_magnification_is_anisotropic(experiment_dataset)

    if current_size is None:
        # The resident drivers score RELION's window at every size, the box included
        # (window_at_box below), so the full box is an explicit current size here.
        current_size = int(experiment_dataset.image_shape[0])
    wsum_current_size = current_size if wsum_current_size is None else int(wsum_current_size)
    if wsum_current_size != current_size and (tilt is not None or full_grid_engine is not None):
        raise NotImplementedError("a weighted-sum size above the scoring size (--strict_highres_exp) is SPA resident only")
    try:
        (
            mstep_current_size,
            n_half,
            window_spec_kwargs,
            budget_window_spec,
            device_memory_bytes,
            precision_policy,
        ) = pass2_window_setup(
            image_shape,
            current_size=current_size,
            reconstruction_current_size=reconstruction_current_size,
            half_spectrum_scoring=half_spectrum_scoring,
            square_window=square_window,
            relion_firstiter_score_mode=relion_firstiter_score_mode,
            use_exact_relion_gaussian=use_exact_relion_gaussian,
            use_float64_scoring=use_float64_scoring,
            device_memory_override=engine_tuning.device_memory_bytes,
            # RELION's window at every size, including the box (a shape class reaches its
            # box before the reference does): the resident driver never scores a full half.
            window_at_box=True,
            reference_sphere_clip=reconstruction_image_radius is not None or anisotropic_magnification,
        )
    except NotImplementedError as exc:
        # A window the resident scorer does not implement is a configuration gap,
        # reported before any device work like the checks below.
        raise ResidentConfigurationUnsupported(str(exc)) from exc

    accumulate_scale, relion_wavg_atomic_scale_aa = resident_pass_tables.resident_scale_and_triplet(
        group_ids=group_ids, accumulate_noise=accumulate_noise
    )
    resolved_spectrum_norm, relion_exact_bpref_operands = _resident_wavg_arithmetic(
        source_faithful_spectrum_norm=source_faithful_spectrum_norm,
    )
    if not relion_exact_bpref_operands and dataset_needs_exact_ctf(experiment_dataset):
        # CTF-premultiplied images, even Zernike terms and magnification (a K>1 pass):
        # only the exact operand family is built from relax's exact RELION CTF rows.
        relion_exact_bpref_operands = True
    # A fresh K=1 pass scores its fine diff2 in RELION's native FFT units, on
    # the compact engine's condition (cf218a8): its fresh guard is the
    # unresolved ``source_faithful_spectrum_norm`` argument, and RELION's
    # RFLOAT CTF operand exists exactly when the exact BPref operands are on.
    # Only the score operands change (the score-cache rows, the score image
    # and corr_img); the reconstruction and noise operands keep RECOVAR units.
    relion_native_fine_units = relion_native_fine_units_enabled(
        fresh_k1_guard=bool(source_faithful_spectrum_norm),
        use_exact_relion_gaussian=use_exact_relion_gaussian,
        use_float64_scoring=use_float64_scoring,
        has_ctf_rfloat=relion_exact_bpref_operands,
    ) and not (firstiter_cc and tilt is not None)  # a tilt CC pass translates its score image (resident_tilts)
    native_fft_size = int(np.prod(image_shape))

    projection_cache_enabled = projection_cache_enabled_for_pass(
        fine_rotations_override=fine_rotations_override,
        dump_pass2_operands=False,
        requested=engine_tuning.pass2_projection_cache,
    )
    if full_grid_engine is None:
        resident_configuration.require_resident_production_configuration(
        relion_x_half_mstep=relion_x_half_mstep,
        relion_exact_fine_gaussian=relion_exact_fine_gaussian,
        relion_exact_fine_normalized_cc=relion_exact_fine_normalized_cc,
        relion_firstiter_score_mode=relion_firstiter_score_mode,
        use_float64_scoring=use_float64_scoring,
        relion_firstiter_winner_take_all=relion_firstiter_winner_take_all,
        return_score_log_z_only=return_score_log_z_only,
        accumulate_noise=accumulate_noise,
        mstep_subtract_ctf_projection=mstep_subtract_ctf_projection,
        normalization_log_z=normalization_log_z,
        normalization_other_score_log_z=normalization_other_score_log_z,
        relion_f32_normalization_sum_weight=relion_f32_normalization_sum_weight,
        relion_coarse_hard_assignment=relion_coarse_hard_assignment,
        relion_coarse_max_posterior=relion_coarse_max_posterior,
        oversampling_order=oversampling_order,
        fine_rotations_override=fine_rotations_override,
        fine_rotation_parent_override=fine_rotation_parent_override,
        use_window=budget_window_spec.use_window,
        projection_cache_available=projection_cache_enabled,
        relion_wavg_atomic_scale_aa=relion_wavg_atomic_scale_aa,
        relion_projector_texture=relion_projector_texture,
        requested_diagnostics=engine_tuning.resident_diagnostics,
        )
        resident_configuration.require(
            bool(use_relion_f32_fine_posterior) or firstiter_cc,
            "the RELION float32 fine posterior is the segmented kernel's contract "
            "(the --firstiter_cc pass takes the winner instead)",
        )
        resident_configuration.require(
            bool(relion_fine_mstep_prune) or bool(relion_x_half_mstep),
            "the resident M-step reconstructs from RELION's pruned fine weights",
        )
    # ---- accumulator layout (identical to the compact engine) -------------
    volume_current_size, recon_volume_shape, mstep_max_r = resident_pass_tables.reconstruction_volume(
        volume_shape,
        mstep_current_size=mstep_current_size,
        reconstruction_volume_current_size=reconstruction_volume_current_size,
        reconstruction_padding_factor=reconstruction_padding_factor,
        reconstruction_image_radius=reconstruction_image_radius,
        anisotropic_magnification=anisotropic_magnification,
    )
    recon_accum_shape = half_volume_accumulator_shape(recon_volume_shape)
    recon_volume_size = int(np.prod(recon_accum_shape))
    recon_y_accum_dtype, recon_ctf_accum_dtype = relion_x_half_mstep_accumulator_dtypes(
        experiment_dataset.dtype,
        use_relion_x_half_mstep=True,
        mstep_double=scoring_overrides.x_half_mstep_double,
    )
    logger.info(
        "Resident pass-2 RELION x-half current-size BPref accumulator shape: "
        "volume_shape=%s score_current_size=%s wsum_current_size=%s model_current_size=%s padding_factor=%s "
        "recon_volume_shape=%s half_accum_shape=%s voxels=%d",
        tuple(volume_shape),
        current_size,
        wsum_current_size,
        mstep_current_size,
        reconstruction_padding_factor,
        tuple(recon_volume_shape),
        tuple(recon_accum_shape),
        recon_volume_size,
    )

    # ---- projection volumes, one per class --------------------------------
    use_relion_projector = relion_projector_half is not None
    class_projector_halves = [None] * n_classes
    if use_relion_projector:
        if relion_projector_r_max is None:
            raise ValueError("relion_projector_r_max is required when relion_projector_half is provided")
        # A K-class stack is split on the host, so only one class's slab at a
        # time is converted on the device.
        stacked_halves = relion_projector_half
        class_projector_halves = []
        for class_index in range(n_classes):
            relion_projector_half = jnp.asarray(
                stacked_halves if classes is None else stacked_halves[class_index]
            )
            # RELION projects through a float32 texture (AccProjector::setMdlData).
            # Left complex128, this driver's projections fell back to the vmapped
            # JAX projector: different values (os1 iteration 1: hard assignments
            # 97.7% equal) and twice the iteration time.
            if not use_float64_scoring and relion_projector_half.dtype == jnp.complex128:
                relion_projector_half = relion_projector_half.astype(jnp.complex64)
            class_projector_halves.append(relion_projector_half)
        del stacked_halves, relion_projector_half
    class_volumes = [volume] if classes is None else [volume[k] for k in range(n_classes)]
    if projection_padding_factor > 1 and not use_relion_projector:
        from relax.reconstruction.relion_functions import pad_volume_for_projection

        class_means_for_proj = []
        for class_volume in class_volumes:
            mean_for_proj, proj_volume_shape = pad_volume_for_projection(
                class_volume,
                volume_shape,
                projection_padding_factor,
                do_gridding_correction=do_gridding_correction,
                current_size=mstep_current_size,
            )
            class_means_for_proj.append(mean_for_proj)
    else:
        class_means_for_proj = class_volumes
        proj_volume_shape = volume_shape

    # ---- fine translations and priors -------------------------------------
    translations_source_np = np.asarray(translations)
    translations_np = np.asarray(translations_source_np, dtype=precision_policy.score_real_dtype)
    if translation_step is None:
        translation_step = infer_translation_step(translations_np)
    if fine_translations_override is None and fine_translation_parent_override is None:
        fine_translations_source, fine_translation_parent = get_oversampled_translation_grid(
            translations_source_np,
            translation_step,
            oversampling_order=oversampling_order,
        )
        fine_translations = np.asarray(
            fine_translations_source, dtype=precision_policy.score_real_dtype
        )
        fine_translation_parent = np.asarray(fine_translation_parent, dtype=np.int32)
    elif fine_translations_override is not None and fine_translation_parent_override is not None:
        fine_translations_source = np.asarray(fine_translations_override)
        fine_translations = np.asarray(
            fine_translations_source, dtype=precision_policy.score_real_dtype
        )
        fine_translation_parent = np.asarray(fine_translation_parent_override, dtype=np.int32)
    else:
        raise ValueError(
            "fine_translations_override and fine_translation_parent_override must be provided together",
        )
    n_fine_trans = int(fine_translations.shape[0])
    if image_translations is not None and (
        tilt is not None
        or full_grid_engine is not None
        or np.shape(image_translations) != (n_images, 2)
        or n_fine_trans != 1
        or np.any(np.asarray(fine_translations) != 0)
    ):
        raise ValueError(
            "image_translations are one [n_images, 2] translation per single-particle image, "
            "scored on the one zero fine translation"
        )

    translation_prior_centers_np = validate_translation_prior_centers(
        translation_prior_centers,
        n_images=n_images,
        n_dims=translations_np.shape[1],
    )
    fine_translation_prior_2d = fine_translation_log_prior_2d(
        translation_log_prior,
        fine_translation_parent,
        n_images=n_images,
        n_fine_trans=n_fine_trans,
        dtype=precision_policy.score_real_dtype,
    )

    if full_grid_engine is not None:
        from relax.fine_pass.full_grid import DenseGemmPreparedState

        if tilt is not None:
            raise NotImplementedError("dense GEMM global pass currently requires SPA")
        if any(support is not None and any(item is not None for item in support)
               for support in class_supports):
            raise ValueError("dense GEMM global pass requires the complete unpruned expanded grid")
        if use_float64_scoring or not relion_x_half_mstep:
            raise ValueError("dense GEMM global pass requires float32 scoring and RELION x-half BPref")
        if normalization_log_z is not None or normalization_other_score_log_z is not None or relion_f32_normalization_sum_weight is not None:
            raise ValueError("dense GEMM global pass does not consume external/coarse normalizers")
        return full_grid_engine(DenseGemmPreparedState(
            dataset=experiment_dataset,
            class_volumes=tuple(class_volumes),
            class_projector_halves=tuple(class_projector_halves),
            class_rotation_priors=tuple(class_rotation_priors),
            fine_rotations=fine_rotations_override,
            fine_mstep_rotations=(fine_rotations_override if fine_mstep_rotations_override is None
                                  else fine_mstep_rotations_override),
            fine_rotation_parent=fine_rotation_parent_override,
            fine_translations_source=fine_translations_source,
            fine_translations=fine_translations,
            fine_translation_parent=fine_translation_parent,
            fine_translation_prior=fine_translation_prior_2d,
            translation_prior_centers=translation_prior_centers_np,
            noise_variance_half=noise_variance,
            n_coarse_rot=n_coarse_rot,
            nside_level=nside_level,
            oversampling_order=oversampling_order,
            random_perturbation=random_perturbation,
            relion_parent_execution_order=use_relion_f32_fine_posterior,
            current_size=int(current_size), mstep_current_size=int(mstep_current_size),
            n_half=n_half, window_spec_kwargs=window_spec_kwargs, engine_tuning=engine_tuning,
            disc_type=disc_type, projector_r_max=relion_projector_r_max,
            projection_padding_factor=projection_padding_factor,
            reconstruction_padding_factor=reconstruction_padding_factor,
            reconstruction_volume_current_size=reconstruction_volume_current_size,
            reconstruction_image_radius=reconstruction_image_radius,
            image_corrections=image_corrections, scale_corrections=scale_corrections,
            image_pre_shifts=image_pre_shifts, group_ids=group_ids,
            scale_correction_group_count=scale_correction_group_count,
            scale_correction_data_vs_prior=scale_correction_data_vs_prior,
            optics_group_ids=optics_group_ids,
            reconstruction_group_ids=reconstruction_group_ids,
            reconstruction_group_count=reconstruction_group_count,
            score_with_masked_images=score_with_masked_images,
            half_spectrum_scoring=half_spectrum_scoring, square_window=square_window,
            accumulate_noise=accumulate_noise,
            use_exact_relion_gaussian=use_exact_relion_gaussian,
            source_faithful_spectrum_norm=resolved_spectrum_norm,
            relion_wavg_atomic_scale_aa=relion_wavg_atomic_scale_aa,
            accumulate_scale=accumulate_scale,
            relion_exact_bpref_operands=relion_exact_bpref_operands,
            relion_native_fine_units=relion_native_fine_units,
            relion_firstiter_score_mode=relion_firstiter_score_mode,
            mstep_subtract_ctf_projection=mstep_subtract_ctf_projection,
            include_unweighted_norm_high_shell=include_unweighted_norm_high_shell,
            relion_translation_angle_scale=relion_translation_angle_scale,
            precision_policy=precision_policy,
            symmetry_label=symmetry_label,
        ))

    # ---- per-image hypotheses and candidate tables, one per class ---------
    # A K-class table joins the classes' own tables in RELION's class-major
    # hidden space (merge_class_tables; docs/development/resident_segments.md).
    # The classes build side by side (map_over_classes); prep_s and table_s
    # sum the per-class seconds, which can exceed the wall of the build. Tables
    # from the device-compacted CSR are built per image block in the chunk loop
    # (resident_table_blocks.candidate_table_blocks).
    tables, prep_s, table_s = resident_table_blocks.candidate_table_blocks(
        class_supports,
        class_rotation_priors,
        whole=relion_f32_normalization_sum_weight is not None,
        reconstruction_group_ids=reconstruction_group_ids,
        reconstruction_group_count=reconstruction_group_count,
        n_images=n_units,
        n_coarse_rot=n_coarse_rot,
        n_coarse_trans=n_coarse_trans,
        nside_level=nside_level,
        oversampling_order=oversampling_order,
        n_fine_trans=n_fine_trans,
        fine_translation_parent=fine_translation_parent,
        random_perturbation=random_perturbation,
        fine_source_eulers_override=fine_source_eulers_override,
        fine_rotations_override=fine_rotations_override,
        fine_mstep_rotations_override=fine_mstep_rotations_override,
        fine_rotation_parent_override=fine_rotation_parent_override,
        use_relion_f32_fine_posterior=use_relion_f32_fine_posterior,
        dtype=precision_policy.score_real_dtype,
        symmetry_label=symmetry_label,
        coarse_rotation_ids=coarse_rotation_ids,
        per_image_rotation_log_prior=unit_rotation_log_prior,
    )
    table_t0 = time.time() - table_s  # the plan log's table+plan time includes the build
    coarse_reuse = _coarse_normalization_reuse(
        # The reuse is a whole-pass (one-block) table; it reads every image's rows.
        tables.block_tables(0) if relion_f32_normalization_sum_weight is not None else tables,
        relion_f32_normalization_sum_weight=relion_f32_normalization_sum_weight,
        relion_coarse_hard_assignment=relion_coarse_hard_assignment,
        relion_coarse_max_posterior=relion_coarse_max_posterior,
        fine_rotation_parent=fine_rotation_parent_override,
        fine_translation_parent=fine_translation_parent,
        max_posterior_dtype=precision_policy.score_real_dtype,
        relion_coarse_class_assignment=relion_coarse_class_assignment,
    )

    # ---- window / weights / lookups (unchanged) ---------------------------
    # Tilt passes keep RELION's logical window sizes: their scoring and M-step (resident_tilts) do not
    # take a stable window's logical bounds.
    # The spec's current size: the physical class with stable windows, RELION's otherwise.
    stable_window_plan, stable_window_spec, program_current_size = resident_stable_windows.pass_stable_windows(
        image_shape,
        current_size=current_size,
        wsum_current_size=wsum_current_size,
        mstep_current_size=mstep_current_size,
        n_half=n_half,
        square_window=square_window,
        window_spec_kwargs=window_spec_kwargs,
        firstiter_cc=firstiter_cc,
        requested=tilt is None and engine_tuning.stable_windows,
        quantum=engine_tuning.stable_window_quantum,
        log_label="Resident pass-2",
    )
    if stable_window_plan is not None:
        budget_window_spec = stable_window_spec
    window_setup = sparse_pass2_window_setup(
        experiment_dataset,
        disc_type=disc_type,
        image_shape=image_shape,
        current_size=current_size,
        n_half=n_half,
        mstep_current_size=mstep_current_size,
        square_window=square_window,
        window_spec_kwargs=window_spec_kwargs,
        use_relion_x_half_mstep=True,
        log_label="Resident pass-2",
        window_spec_override=stable_window_spec,
    )
    config = window_setup.config
    window_spec = window_setup.window_spec
    window_indices_np = window_setup.window_indices_np
    window_indices = window_setup.window_indices
    recon_window_indices = window_setup.recon_window_indices
    relion_x_half_recon_indices = window_setup.relion_x_half_recon_indices
    windowed_prepare = window_setup.windowed_prepare
    n_windowed = window_setup.n_windowed
    n_recon_windowed = window_setup.n_recon_windowed

    half_weights, half_weights_windowed = pass2_half_weights(
        image_shape,
        window_spec,
        half_spectrum_scoring=half_spectrum_scoring,
        relion_firstiter_score_mode=relion_firstiter_score_mode,
        use_float64_scoring=use_float64_scoring,
        nyquist_column_counting=nyquist_column_counting,
        firstiter_cc_support=firstiter_cc_support,
        current_size=current_size,
    )
    half_weights_windowed, relion_score_full_to_compact = resident_stable_windows.score_weights_and_lookup(
        half_weights_windowed,
        image_shape=image_shape,
        current_size=current_size,
        window_indices_np=window_indices_np,
        plan=stable_window_plan,
    )
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(
        noise_variance, image_shape
    ).squeeze()
    n_optics_groups, optics_groups_np = resident_pass_tables.optics_group_rows(
        noise_variance_half, optics_group_ids, n_images=n_images
    )
    # Tilt images (S4.2) carry 3D translations whose phases are per image (tilt.image_angles); the SPA
    # operand preparation, which never sees a trial shift of a tilt image, gets 2D zeros of the same count.
    spa_fine_translations_source = (
        fine_translations_source if tilt is None else np.zeros((n_fine_trans, 2), dtype=np.float64)
    )
    spa_fine_translations = (
        fine_translations if tilt is None else np.zeros((n_fine_trans, 2), dtype=fine_translations.dtype)
    )
    relion_score_translation_angles = relion_cuda_score_translation_angles_if_available(
        spa_fine_translations_source,
        image_shape,
        enabled=True,
        dtype=np.float64 if use_float64_scoring else np.float32,
        angle_scale=relion_translation_angle_scale,
    )
    if relion_score_translation_angles is None:
        raise ValueError("the resident scoring stage requires RELION translation angles")
    translation_phases_half = (
        None if windowed_prepare else half_translation_phase_table(spa_fine_translations, image_shape)
    )

    n_shells = image_shape[0] // 2 + 1
    (
        shell_indices_half,
        shell_indices_noise,
        noise_variance_for_noise,
        relion_wavg_rectangle,
        logical_rect_pixels,
        logical_recon_pixels,
    ) = resident_pass_tables.resident_noise_tables(
        noise_variance_half,
        window_spec=window_spec,
        window_indices=window_indices,
        window_indices_np=window_indices_np,
        recon_window_indices=recon_window_indices,
        n_recon_windowed=n_recon_windowed,
        stable_window_plan=stable_window_plan,
        image_shape=image_shape,
        current_size=current_size,
        wsum_current_size=wsum_current_size,
        mstep_current_size=mstep_current_size,
    )
    n_rect = int(relion_wavg_rectangle.centered_indices.size)
    # The BPref accumulators and the adjoint are keyed on the physical class too.
    program_volume_current_size, program_recon_volume_shape, program_mstep_max_r = (
        resident_stable_windows.physical_reconstruction_volume(
            stable_window_plan,
            volume_shape=volume_shape,
            volume_current_size=volume_current_size,
            reconstruction_volume_current_size=reconstruction_volume_current_size,
            recon_volume_shape=recon_volume_shape,
            mstep_max_r=mstep_max_r,
            reconstruction_image_radius=reconstruction_image_radius,
            reconstruction_padding_factor=reconstruction_padding_factor,
            anisotropic_magnification=anisotropic_magnification,
            quantum=engine_tuning.stable_window_quantum,
        )
    )
    program_recon_volume_size = int(np.prod(half_volume_accumulator_shape(program_recon_volume_shape)))
    # One x-half BPref pair per accumulator slot: RELION's BPref[iclass], and for
    # VDAM's pseudo-halfsets BPref[iclass + half * nr_classes]. They exist before
    # the pass reads free device memory (projection cache, resident operand and
    # chunk budgets), so every budget sees them; allocated after the chunk
    # budget they were not counted, up to 23 GiB at EMPIAR-10202's full box.
    # A K-class pass whose sums fit carries per-projection row sums instead
    # (projection_sums_bytes); the volumes are then built after the chunk loop.
    # K=1 keeps the per-row backprojection: it measured no gain (K1 50k:
    # 1095 -> 1092 s), and its cold-start parity cases carry the
    # reassociation into the trajectory (fast tier k1_os1_coldstart_standalone
    # half-1 FSC-AUC 1.000000 -> 0.998679).
    n_fine_rot = int(np.asarray(fine_rotations_override).shape[0])
    sums_bytes = resident_projections.projection_sums_bytes(
        n_slots=int(tables.n_slots),
        n_fine_rot=n_fine_rot,
        n_recon_pixels=int(n_recon_windowed),
        y_dtype=recon_y_accum_dtype,
        ctf_dtype=recon_ctf_accum_dtype,
    )
    # What the JAX allocator can still hand out, not the device's physical free memory: under a pool
    # limit (XLA_PYTHON_CLIENT_MEM_FRACTION, or a smaller card's pool) the physical reading counted an
    # 80 GB card's 62 GiB and chose 3.76 GiB of sums at a 16 GB pool, which then ran out preparing the
    # half's operands (relax#41, EMPIAR-10073 box 380 Class3D K3).
    free_bytes = current_device_available_bytes()
    # A tilt chunk backprojects each (image, row) with its own matrix (resident_tilts), never per projection.
    presum_adjoint = (
        n_classes > 1
        and tilt is None
        and free_bytes is not None
        and sums_bytes <= resident_projections.PRESUM_ADJOINT_FREE_FRACTION * float(free_bytes)
        and not resident_projections.projection_sums_displace_resident_operands(
            sums_bytes,
            operand_max_bytes=engine_tuning.resident_operand_max_bytes,
            resident_operands=engine_tuning.resident_operands,
            n_images=n_images,
            n_windowed=n_windowed,
            n_recon_windowed=n_recon_windowed,
            n_rect=n_rect,
            n_shells=n_shells,
            n_fine_trans=n_fine_trans,
            precision_policy=precision_policy,
            resolved_spectrum_norm=resolved_spectrum_norm,
            premultiplied_ctf=ctf_premultiplied_pass,
            image_shape=image_shape,
        )
    )
    accumulator_shape = (n_fine_rot, int(n_recon_windowed)) if presum_adjoint else (program_recon_volume_size,)
    # Every slot's two accumulators are zeroed by one program per shape class.
    zero_accumulators = resident_mstep_blocks.zero_block_partials(
        ((tuple(accumulator_shape), jnp.dtype(recon_y_accum_dtype)),) * tables.n_slots
        + ((tuple(accumulator_shape), jnp.dtype(recon_ctf_accum_dtype)),) * tables.n_slots
    )()
    Ft_y_total = tuple(zero_accumulators[:tables.n_slots])
    Ft_ctf_total = tuple(zero_accumulators[tables.n_slots:])
    del zero_accumulators
    # Rows backprojected one by one go through a partial pair (relax#65); projection sums are backprojected
    # once per pass and need none. Allocated here so that the chunk budget below counts it.
    row_partial = (
        None
        if presum_adjoint
        else allocate_row_partial(
            program_recon_volume_size,
            recon_y_accum_dtype,
            recon_ctf_accum_dtype,
            stock_relion_direct_accumulation=scoring_overrides.stock_relion_direct_accumulation,
        )
    )
    logger.info(
        "Resident pass-2 M-step adjoint: %s (per-projection sums %.2f GiB, %s GiB available to the allocator)",
        "per-projection sums, one backprojection per pass" if presum_adjoint else "per-row backprojection",
        sums_bytes / float(1024**3),
        "unknown" if free_bytes is None else f"{float(free_bytes) / float(1024**3):.2f}",
    )
    scale_pixel_mask_rect_np = resident_pass_tables.scale_pixel_masks_rect(
        scale_correction_data_vs_prior,
        n_classes=n_classes,
        shell_indices_noise=shell_indices_noise,
        n_shells=n_shells,
        relion_wavg_rectangle=relion_wavg_rectangle,
    )
    if n_classes == 1:
        scale_pixel_mask_rect_np = scale_pixel_mask_rect_np[0]
    group_ids_np, n_scale_groups = prepare_scale_correction_groups(
        group_ids, scale_correction_group_count, n_images=n_images,
    )

    # ---- projection cache (same admission and build as the compact engine) -
    # K>1 caches every class's fine grid, class k at k * n_fine_rot; when the
    # significant coarse parents' children are far fewer, only they are cached
    # (significant_projection_slots), class k at k * capacity.
    projection_slots = None
    if tilt is None and not presum_adjoint and coarse_reuse is None and coarse_rotation_ids is None:
        projection_slots = resident_projections.significant_projection_slots(
            class_supports,
            n_images=n_units,
            n_coarse_rot=n_coarse_rot,
            n_coarse_trans=n_coarse_trans,
            n_fine_rot=n_fine_rot,
            children=fine_rotation_children(
                n_coarse_rot=n_coarse_rot,
                nside_level=nside_level,
                oversampling_order=oversampling_order,
                random_perturbation=random_perturbation,
                fine_rotation_parent_override=fine_rotation_parent_override,
                symmetry_label=symmetry_label,
            ),
        )
    cache_rows_per_class = n_fine_rot if projection_slots is None else int(projection_slots.capacity)
    n_projections = resident_projections.projection_cache_rows(n_classes, n_fine_rot, projection_slots)
    (
        _projection_complex_dtype,
        _projection_budget_pixels,
        max_projected_rotations_per_projection_call,
    ) = pass2_projection_budget(
        jnp.asarray(class_means_for_proj[0]).dtype,
        precision_policy,
        n_half=n_half,
        use_relion_projector=use_relion_projector,
        budget_window_spec=budget_window_spec,
        device_memory_bytes=device_memory_bytes,
        include_abs2=False,
        max_projected_rotations_override=engine_tuning.max_projected_rotations,
        projection_max_bytes=engine_tuning.projection_max_bytes,
    )
    transient_projection_bytes = projection_cache_transient_bytes(
        n_projections,
        n_windowed,
        projection_complex_dtype=precision_policy.score_complex_dtype,
        include_abs2=False,
    ) + projection_cache_transient_bytes(
        n_projections,
        n_recon_windowed,
        projection_complex_dtype=precision_policy.score_complex_dtype,
        include_abs2=True,
    )
    # Union projection cache: one row per rotation holding the score window and
    # the reconstruction window together, from which the chunk programs take
    # each window and form |recon|^2. The two windows differ only by RELION's DC
    # pixel and the rounded shell rim, so the union is a little over a third of
    # the three-cache bytes: the full EMPIAR-10097 hp3 cache (294912 rotations
    # at current size 126) is 14.4 GiB instead of 35.5 GiB and fits instead of
    # streaming its projections per chunk. The values are the same gathers of
    # the same projection. Streamed chunks keep their three chunk-local caches.
    union_indices = union_score_take = union_recon_take = None
    cache_projection_bytes = transient_projection_bytes
    if not firstiter_cc and not use_float64_scoring:
        score_indices_np = np.asarray(window_indices, dtype=np.int64)
        recon_indices_np = np.asarray(recon_window_indices, dtype=np.int64)
        union_indices_np = np.union1d(score_indices_np, recon_indices_np)
        # Cast on the host: a device cast of the int64 indices was a program per size.
        union_indices = jnp.asarray(union_indices_np.astype(np.int32))
        union_score_take = jnp.asarray(np.searchsorted(union_indices_np, score_indices_np).astype(np.int32))
        union_recon_take = jnp.asarray(np.searchsorted(union_indices_np, recon_indices_np).astype(np.int32))
        cache_projection_bytes = projection_cache_transient_bytes(
            n_projections,
            int(union_indices_np.size),
            projection_complex_dtype=precision_policy.score_complex_dtype,
            include_abs2=False,
        )
    max_projection_cache_bytes = projection_cache_max_bytes_for_pass(
        device_memory_bytes, override=engine_tuning.projection_max_bytes,
    )
    projection_kwargs = projection_kwargs_for_relion_score_window(
        window_spec.projection_kwargs(return_abs2=False),
        use_relion_projector=use_relion_projector,
        current_size=current_size,
    )
    projection_kwargs["mask_current_image_disk"] = bool(projection_mask_current_image_disk)
    projection_kwargs["texture_interp_default"] = scoring_overrides.projector_texture_interp
    # Stable windows also project at the physical class (pass_capacity_projection).
    projection_halves, projection_r_max, capacity_projection_kwargs, capacity_logical_output_size = (
        resident_stable_windows.pass_capacity_projection(
            stable_window_plan,
            use_relion_projector=use_relion_projector,
            class_projector_halves=class_projector_halves,
            relion_projector_r_max=relion_projector_r_max,
            projection_padding_factor=projection_padding_factor,
            projection_kwargs=projection_kwargs,
        )
    )
    if capacity_logical_output_size is not None:
        projection_kwargs["projector_output_size"] = int(stable_window_plan.physical_current_size)
    # The fixed-radius projections stage each class's slab into a texture once, kept from pass to pass and
    # refilled (projection_blocks.open_pass_capacity_texture), not per projection call: a real tilt-series final
    # iteration projected every slot block through its own texture, about 45 CUDA array allocations and
    # stagings per chunk with the GPU idle. The same texels and kernel project.
    class_textures = [None] * len(projection_halves)
    if use_relion_projector and not capacity_projection_kwargs:
        class_textures = [
            projection_blocks.open_pass_capacity_texture(
                half,
                relion_projector_r_max=projection_r_max,
                projection_padding_factor=projection_padding_factor,
                projection_kwargs=projection_kwargs,
                class_index=k,
                kept_only=True,
            )
            for k, half in enumerate(projection_halves)
        ]
    residual_sgd_rows = (
        _padded_sgd_residual_rows(_sgd_residual_rows(
            recon_window_indices,
            image_shape=image_shape,
            projector_output_size=(
                projection_kwargs.get("projector_output_size")
                if capacity_logical_output_size is None
                else capacity_logical_output_size
            ),
            r_max=relion_projector_r_max,
            rotations=np.asarray(fine_rotations_override, dtype=np.float64),
        ))
        if mstep_subtract_ctf_projection and use_relion_projector
        else None
    )
    residual_sgd_cache = residual_sgd_take = None

    fine_grid = jnp.asarray(fine_rotations_override, dtype=precision_policy.score_real_dtype)
    # The cache's rotations, class-major: every class's fine grid, or the cached slots' rotations.
    cache_grid = (
        fine_grid
        if projection_slots is None
        else fine_grid[jnp.asarray(projection_slots.slot_projection % n_fine_rot, dtype=jnp.int32)]
    )

    def class_cache_rotations(class_index):
        if projection_slots is None:
            return fine_grid
        start = int(class_index) * cache_rows_per_class
        return cache_grid[start : start + cache_rows_per_class]

    # The RELION projector computes only the windows' pixels, not full half rows.
    fine_window_union = (
        resident_stable_windows.pass_window_union(
            window_indices,
            recon_window_indices,
            image_shape=image_shape,
            projector_output_size=int(projection_kwargs["projector_output_size"]),
            capacity_logical_output_size=capacity_logical_output_size,
        )
        if projection_kwargs.get("projector_output_size") is not None
        else None
    )

    def project_fine_rotations(
        rotations, class_index=0, n_rows=None, *, outputs=None, output_row_ids=None, finalize=True
    ):
        """(score, recon, |recon|^2) projections of ``rotations``, as the cache holds them.

        ``n_rows`` makes them that many rows long, zero past the rotations.
        ``outputs`` and ``output_row_ids`` write into earlier arrays at those rows
        (:func:`project_rows_by_class`); ``finalize`` applies the cache's units to
        the whole arrays, once, on the call that completes them.
        """

        score, recon, recon_abs2 = compute_sparse_pass2_windowed_projections_block(
            class_means_for_proj[class_index],
            jnp.asarray(rotations, dtype=precision_policy.score_real_dtype),
            image_shape,
            proj_volume_shape,
            disc_type,
            score_indices=window_indices,
            recon_indices=recon_window_indices,
            max_projected_rotations=projection_cache_build_max_rotations_per_call(
                max_projected_rotations_per_projection_call,
                int(np.asarray(rotations).shape[0]),
                engine_tuning.max_projected_rotations,
            ),
            output_complex_dtype=precision_policy.score_complex_dtype,
            output_abs2_dtype=precision_policy.score_real_dtype,
            relion_projector_half=projection_halves[class_index],
            relion_projector_r_max=projection_r_max,
            projection_padding_factor=projection_padding_factor,
            relion_projector_capacity_texture=class_textures[class_index],
            window_union=fine_window_union,
            output_rows=n_rows,
            outputs=outputs,
            output_row_ids=output_row_ids,
            **capacity_projection_kwargs,
            **projection_kwargs,
        )
        if not finalize:
            return score, recon, recon_abs2
        recon, recon_abs2 = precision_policy.cast_local_noise_projection_scores(recon, recon_abs2)
        if relion_native_fine_units:
            # Score rows only, for the cached and the streamed paths alike; the
            # recon rows feed the M-step and noise sums in RECOVAR units.
            score = resident_block_programs.relion_native_fine_units_in_place(score, native_fft_size)
        return score, recon, recon_abs2

    def project_ids(ids, n_rows):
        """Projections of the host projection ids ``class * n_fine_rot + rotation``, in order.

        The arrays are ``n_rows`` long, zero past the ids.

        With K>1 each class projects its own ids from its own reference into
        the one set of arrays (:func:`project_rows_by_class`); a class's call is
        padded to :func:`class_call_length` so the projection programs see few
        distinct lengths.
        """

        ids = np.asarray(ids, dtype=np.int64)
        if n_classes == 1:
            return project_fine_rotations(fine_grid[jnp.asarray(ids, dtype=jnp.int32)], n_rows=int(n_rows))
        return project_rows_by_class(
            project_fine_rotations,
            fine_grid[jnp.asarray(ids % n_fine_rot, dtype=jnp.int32)],
            ids // n_fine_rot,
            n_rows=int(n_rows),
        )

    # The whole fine grid is cached when it fits. At healpix order 3 and a real
    # current size it does not (294912 rotations at 136 px is ~40 GiB), so each
    # chunk projects its own distinct fine rotations instead, as RELION projects
    # each particle's significant orientations. The projections are the same
    # arrays gathered through a chunk-local slot; see stream_chunk_projections.
    projection_bytes_per_rotation = transient_projection_bytes / float(max(n_projections, 1))
    # Both the whole-grid cache and the chunk-local caches are sized from what
    # the allocator can still hand out now, after reserving the half's resident
    # operands (allocated later): a fraction of the device total alone let the
    # whole-grid cache take memory that earlier iterations or a capped allocator
    # did not have (K=1 10097 10k at hp3).
    physical_free_bytes = device_free_memory_bytes()
    allocator_free_bytes = jax_allocator_free_memory_bytes()
    pool_free_bytes = jax_allocator_pool_free_bytes()
    # The operands are reserved only when the admission below would take them
    # at this reading: operands that stay per chunk are never allocated, and
    # reserving them anyway left a 0.38 GiB chunk budget at EMPIAR-10202
    # iteration 2 (box 800, bigbox case).
    reserved_operand_bytes = 0
    # A tilt pass's CC iteration translates each image's resident CC operands per image slot (resident_tilts).
    if engine_tuning.resident_operands and (not firstiter_cc or tilt is not None):
        operand_bytes, operand_peak_bytes = resident_projections.resident_half_operand_sizes(
            n_images=n_images,
            n_windowed=n_windowed,
            n_recon_windowed=n_recon_windowed,
            n_rect=n_rect,
            n_shells=n_shells,
            n_fine_trans=n_fine_trans,
            precision_policy=precision_policy,
            resolved_spectrum_norm=resolved_spectrum_norm,
            premultiplied_ctf=ctf_premultiplied_pass,
            image_shape=image_shape,
        )
        if resident_projections.resident_operands_fit(
            operand_peak_bytes,
            device_available_bytes(physical_free_bytes, allocator_free_bytes, pool_free_bytes),
            engine_tuning.resident_operand_max_bytes,
        ):
            reserved_operand_bytes = operand_bytes
    stream_projection_budget_bytes = resident_projections.stream_projection_budget_bytes(
        max_projection_cache_bytes,
        physical_free_bytes=physical_free_bytes,
        allocator_free_bytes=allocator_free_bytes,
        pool_free_bytes=pool_free_bytes,
        reserved_bytes=reserved_operand_bytes,
    )
    # Tilt images project every (image, rotation) pair per chunk (resident_tilts.run_tilt_chunk).
    stream_projections = tilt is not None or not projection_cache_fits_budget(
        cache_projection_bytes, stream_projection_budget_bytes
    )
    # Streamed chunks keep the per-chunk operand preparation. Their working set
    # (chunk-local projection caches and the posterior program's temporaries)
    # is not in the chunk estimate, and next to 21.5 GiB of resident operands
    # the full EMPIAR-10097 hp3 pass (65000 images, current size 126) could not
    # allocate 9.65 GiB; the per-chunk preparation ran that pass
    # at cd26a5e. When the reservation alone is what makes the whole-grid cache
    # miss, the cache is kept instead of the operands.
    operands_yield_to_cache = False
    # Tilt passes always stream and have no per-chunk operand path: they keep the resident operands.
    if tilt is None and stream_projections and reserved_operand_bytes:
        unreserved_budget_bytes = resident_projections.stream_projection_budget_bytes(
            max_projection_cache_bytes,
            physical_free_bytes=physical_free_bytes,
            allocator_free_bytes=allocator_free_bytes,
            pool_free_bytes=pool_free_bytes,
            reserved_bytes=0,
        )
        if projection_cache_fits_budget(cache_projection_bytes, unreserved_budget_bytes):
            stream_projections = False
            operands_yield_to_cache = True
            stream_projection_budget_bytes = unreserved_budget_bytes
        reserved_operand_bytes = 0
    # The union cache is one array. The free totals above do not promise one block
    # that large (EMPIAR-10345 it13 half 2 could not allocate 20.19 GiB with over
    # 40 GiB free), so it is allocated now, and a pass whose
    # allocator refuses it streams its projections instead. Deciding from the
    # allocator's largest free block instead streamed Class3D K4 100k from
    # iteration 19, a 5.50 GiB cache against a 5.25 GiB block, at 3.5x the
    # iteration time.
    union_cache = None
    if not stream_projections and union_indices is not None:
        union_cache = resident_projections.allocate_projection_cache_blocks(
            n_projections, int(union_indices.shape[0]), precision_policy.score_complex_dtype
        )
        if union_cache is None:
            logger.info(
                "Resident pass-2 streams its projections: the allocator could not hand out the "
                "%.2f GiB whole-grid cache in %d blocks or fewer",
                cache_projection_bytes / float(1024**3),
                resident_projections.MAX_PROJECTION_CACHE_BLOCKS,
            )
            stream_projections = True
            operands_yield_to_cache = False
        elif isinstance(union_cache, tuple):
            logger.info(
                "Resident pass-2 holds its %.2f GiB whole-grid cache as %d row blocks: the allocator "
                "could not hand it out as one",
                cache_projection_bytes / float(1024**3),
                len(union_cache),
            )
    if residual_sgd_rows is not None and (stream_projections or union_indices is None):
        raise NotImplementedError(
            "VDAM's subtracted reference on an image window wider than the model sphere (an optics "
            "group on a coarser grid) is projected with the SGD kernel's rows only beside the union "
            "projection cache; this pass streams its projections or keeps three caches"
        )
    stream_keeps_chunk_operands = tilt is None and (stream_projections or operands_yield_to_cache)
    if stream_projections:
        # Streamed chunks project their own rows' rotations by projection id.
        projection_slots = None
        score_cache = recon_cache = recon_abs2_cache = None
        union_indices = union_score_take = union_recon_take = None
        logger.info(
            "Resident pass-2 projections are streamed per chunk: the %d-rotation cache "
            "would take %.2f GiB against a %.2f GiB budget (cache share %.2f GiB) "
            "at %.1f KiB per rotation (physical free %s, allocator free %s, "
            "pool free %s, reserved operands %.2f GiB)",
            n_projections,
            cache_projection_bytes / float(1024**3),
            stream_projection_budget_bytes / float(1024**3),
            max_projection_cache_bytes / float(1024**3),
            projection_bytes_per_rotation / 1024.0,
            "unknown" if physical_free_bytes is None else f"{physical_free_bytes / float(1024**3):.2f} GiB",
            "unknown" if allocator_free_bytes is None else f"{allocator_free_bytes / float(1024**3):.2f} GiB",
            "unknown" if pool_free_bytes is None else f"{pool_free_bytes / float(1024**3):.2f} GiB",
            reserved_operand_bytes / float(1024**3),
        )
    elif union_indices is not None:
        cache_t0 = time.time()
        union_window_union = (
            resident_stable_windows.pass_window_union(
                union_indices_np,
                image_shape=image_shape,
                projector_output_size=int(projection_kwargs["projector_output_size"]),
                capacity_logical_output_size=capacity_logical_output_size,
            )
            if projection_kwargs.get("projector_output_size") is not None
            else None
        )
        # Raw union rows: the native-unit division of the score window runs
        # after the chunk's gather (score_resident_chunk), and the recon window
        # keeps RECOVAR units, as in the three-cache build.
        rows_per_call = projection_cache_build_max_rotations_per_call(
            max_projected_rotations_per_projection_call, cache_rows_per_class, engine_tuning.max_projected_rotations,
        ) or cache_rows_per_class

        def project_union_rows(class_index, start, stop):
            return compute_sparse_pass2_windowed_projections_block(
                class_means_for_proj[class_index],
                class_cache_rotations(class_index)[start:stop],
                image_shape,
                proj_volume_shape,
                disc_type,
                score_indices=union_indices,
                recon_indices=None,
                max_projected_rotations=rows_per_call,
                output_complex_dtype=precision_policy.score_complex_dtype,
                relion_projector_half=projection_halves[class_index],
                relion_projector_r_max=projection_r_max,
                projection_padding_factor=projection_padding_factor,
                window_union=union_window_union,
                **capacity_projection_kwargs,
                **projection_kwargs,
            )[0]

        score_cache = resident_projections.build_projection_cache_in_place(
            project_union_rows,
            n_classes=n_classes,
            n_rows_per_class=cache_rows_per_class,
            n_pixels=int(union_indices.shape[0]),
            rows_per_call=rows_per_call,
            dtype=precision_policy.score_complex_dtype,
            cache=union_cache,
        )
        del union_cache
        recon_cache = recon_abs2_cache = None
        logger.info(
            "Resident pass-2 projection cache: cached %d fine rotations in %.2fs as one "
            "union window of %d pixels (%.2f GiB; three caches would take %.2f GiB)",
            n_projections,
            time.time() - cache_t0,
            int(union_indices.shape[0]),
            cache_projection_bytes / float(1024**3),
            transient_projection_bytes / float(1024**3),
        )
        if residual_sgd_rows is not None:
            # VDAM subtracts the SGD kernel's projection, which keeps the rows the
            # cache zeroes; those pixels' rows are cached beside it (_sgd_residual_rows).
            sgd_positions, sgd_pixels = residual_sgd_rows
            sgd_pixels_device = jnp.asarray(sgd_pixels)
            sgd_window_union = (
                _padded_window_union(
                    resident_stable_windows.pass_window_union(
                        sgd_pixels.astype(np.int64),
                        image_shape=image_shape,
                        projector_output_size=int(projection_kwargs["projector_output_size"]),
                        capacity_logical_output_size=capacity_logical_output_size,
                    )
                )
                if projection_kwargs.get("projector_output_size") is not None
                else None
            )
            sgd_projection_kwargs = {**projection_kwargs, "relion_kernel": "sgd"}

            def project_sgd_rows(class_index, start, stop):
                return compute_sparse_pass2_windowed_projections_block(
                    class_means_for_proj[class_index],
                    class_cache_rotations(class_index)[start:stop],
                    image_shape,
                    proj_volume_shape,
                    disc_type,
                    score_indices=sgd_pixels_device,
                    recon_indices=None,
                    max_projected_rotations=rows_per_call,
                    output_complex_dtype=precision_policy.score_complex_dtype,
                    relion_projector_half=projection_halves[class_index],
                    relion_projector_r_max=projection_r_max,
                    projection_padding_factor=projection_padding_factor,
                    window_union=sgd_window_union,
                    **capacity_projection_kwargs,
                    **sgd_projection_kwargs,
                )[0]

            residual_sgd_cache = resident_projections.build_projection_cache_in_place(
                project_sgd_rows,
                n_classes=n_classes,
                n_rows_per_class=cache_rows_per_class,
                n_pixels=int(sgd_pixels.size),
                rows_per_call=rows_per_call,
                dtype=precision_policy.score_complex_dtype,
            )
            residual_sgd_take = jnp.asarray(sgd_positions)
    else:
        cache_t0 = time.time()
        if n_classes == 1:
            score_cache, recon_cache, recon_abs2_cache = project_fine_rotations(
                fine_rotations_override if projection_slots is None else class_cache_rotations(0)
            )
        else:
            # Each class's caches are written into the three caches in place,
            # so only one class's caches are ever transient, not a second copy.
            caches = None
            for class_index in range(n_classes):
                class_cache = project_fine_rotations(
                    fine_rotations_override if projection_slots is None else class_cache_rotations(class_index),
                    class_index,
                )
                if caches is None:
                    caches = [
                        jnp.zeros((n_classes * int(a.shape[0]),) + tuple(a.shape[1:]), dtype=a.dtype)
                        for a in class_cache
                    ]
                caches = [
                    resident_projections.write_projection_cache_rows(cache, rows, np.int32(class_index * int(rows.shape[0])))
                    for cache, rows in zip(caches, class_cache, strict=True)
                ]
                del class_cache
            score_cache, recon_cache, recon_abs2_cache = caches
            del caches
        logger.info(
            "Resident pass-2 projection cache: cached %d fine rotations in %.2fs "
            "(estimated transient %.2f GiB)",
            n_projections,
            time.time() - cache_t0,
            transient_projection_bytes / float(1024**3),
        )

    # ---- capacity plan ----------------------------------------------------
    # A chunk gathers one cached row per candidate: the union window's pixels
    # when the cache holds one (cached_block_projections), not the score
    # window's. Counting the score window let a 524288-row chunk past both
    # budgets at current size 108 and its 19.8 GiB gather ran out of memory
    # (union 5008 vs score 3690 pixels).
    gathered_row_pixels = int(n_windowed) if union_indices is None else int(union_indices.shape[0])
    # A row-blocked cache's gather holds the rows taken so far next to the next
    # block's (resident_scoring.cache_rows): two row copies instead of one.
    gathered_row_copies = 2 if isinstance(score_cache, tuple) else 1
    configured_row_ladder = engine_tuning.resident_row_capacities or _DEFAULT_ROW_CAPACITY_LADDER
    row_ladder = configured_row_ladder
    if stream_projections:
        row_ladder = resident_projections.stream_row_capacity_ladder(
            row_ladder,
            bytes_per_rotation=projection_bytes_per_rotation,
            max_projection_bytes=stream_projection_budget_bytes,
        )
    else:
        # The chunk gathers its rows out of the cache into one [rows, pixels]
        # block; at box 800 (10202, current size 304) capacity 131072 needs
        # 35.7 GiB for it. The budget is read again now that the
        # cache is resident; unknown readings do not cap.
        gather_budget_bytes = resident_projections.stream_projection_budget_bytes(
            device_memory_bytes if device_memory_bytes is not None else 1 << 62,
            physical_free_bytes=device_free_memory_bytes(),
            allocator_free_bytes=jax_allocator_free_memory_bytes(),
            pool_free_bytes=jax_allocator_pool_free_bytes(),
            reserved_bytes=reserved_operand_bytes,
        )
        row_ladder = resident_projections.cached_row_capacity_ladder(
            row_ladder,
            bytes_per_row=gathered_row_copies
            * gathered_row_pixels
            * np.dtype(precision_policy.score_complex_dtype).itemsize,
            max_gather_bytes=gather_budget_bytes,
        )
        logger.info(
            "Resident pass-2 cached-path row capacities %s: gather budget %.2f GiB at %.1f KiB per row",
            ",".join(str(v) for v in row_ladder),
            gather_budget_bytes / float(1024**3),
            gathered_row_pixels * np.dtype(precision_policy.score_complex_dtype).itemsize / 1024.0,
        )
    # The chunk's translated arrays depend on its operand family, so the plan
    # counts the family the pass expects: the half's resident operands when
    # they are reserved above. The operand selection below can still fall back
    # to the per-chunk preparation (a re-measured budget, a refused
    # configuration); the pass is then planned again for it.
    n_half_pixels = int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)
    row_ladder_start = row_ladder

    # A tilt unit gathers its S images with one zero translation (S tiles where SPA has T per image) and
    # each of its rows is projected once per image slot; the M-step's translated tiles are sized per
    # chunk in translation blocks (resident_tilts.run_tilt_chunk, mstep_translation_blocks).
    plan_translations = n_fine_trans if tilt is None else int(tilt.slot_capacity)
    # Device bytes of one row's projections (one rotation's; a tilt row has one per image slot).
    projection_row_bytes = (
        resident_projections.STREAM_PEAK_COPIES * int(projection_bytes_per_rotation)
        if stream_projections
        else gathered_row_pixels * np.dtype(precision_policy.score_complex_dtype).itemsize
    )
    tilt_slot_block = None
    # Set by plan_chunks: whether a tilt pass runs its over-budget particles in row blocks, and the block rows of
    # their lean form (resident_tilt_lone) when the full form does not fit either.
    tilt_lone_row_blocks = False
    tilt_lean_lone_rows = None
    if tilt is not None:
        from relax.fine_pass.resident_tilts import (
            TILT_MIN_PARTICLES_PER_CHUNK,
            TILT_TARGET_PARTICLES_PER_CHUNK,
            plan_lean_lone_blocks,
            tilt_amortised_row_classes,
            tilt_capacity_ladders,
            tilt_lone_row_bytes,
            tilt_projection_slot_block,
        )

        row_ladder_start, _ = tilt_capacity_ladders(
            row_ladder_start, (1,), slot_capacity=int(tilt.slot_capacity)
        )
        # Too few particles per chunk at these classes: one larger class, from the configured ladder's top and the
        # streamed projections' budget (tilt_amortised_row_classes). A tilt pass always streams its projections
        # (stream_projections above), so that budget is the one that capped row_ladder.
        assert stream_projections, "a tilt pass streams its projections"
        top_tilt_rows, _ = tilt_capacity_ladders(configured_row_ladder, (1,), slot_capacity=int(tilt.slot_capacity))
        amortised_rows, mean_particle_rows = tilt_amortised_row_classes(
            row_ladder_start,
            tables.row_offsets,
            slot_capacity=int(tilt.slot_capacity),
            row_bytes=projection_row_bytes,
            budget_bytes=stream_projection_budget_bytes,
            max_rows=max(top_tilt_rows),
        )
        if amortised_rows != tuple(row_ladder_start):
            previous_largest = max(row_ladder_start)
            logger.info(
                "Resident pass-2 tilt row classes %s -> %s: particles of %.0f rows on average fill a %d-row chunk with "
                "%.1f particles (fewer than %d; the new class holds %d)",
                ",".join(str(v) for v in row_ladder_start),
                ",".join(str(v) for v in amortised_rows),
                mean_particle_rows,
                previous_largest,
                previous_largest / mean_particle_rows,
                TILT_MIN_PARTICLES_PER_CHUNK,
                TILT_TARGET_PARTICLES_PER_CHUNK,
            )
        row_ladder_start = amortised_rows
        largest_row_class = max(row_ladder_start)
        # A chunk whose rows' projections over all its images do not fit is projected, scored and
        # backprojected a block of image slots at a time (resident_tilts.run_tilt_chunk). A chunk of the row
        # classes takes the largest class's slot block; a particle past them (run alone) takes the largest
        # particle's, which bounds every such chunk's projections (relax#115: sized for all chunks, a 37k-row
        # tied-block particle split every 500-row chunk of the pass into 21 slot blocks).
        tilt_slot_row_bytes = projection_row_bytes * gathered_row_copies
        tilt_slot_budget_bytes = resident_chunk_memory.resident_chunk_budget_bytes(reserved_bytes=reserved_operand_bytes)
        tilt_slot_block = tilt_projection_slot_block(
            largest_row_class,
            slot_capacity=int(tilt.slot_capacity),
            slot_row_bytes=tilt_slot_row_bytes,
            budget_bytes=tilt_slot_budget_bytes,
        )
        tilt_overflow_slot_block = tilt_projection_slot_block(
            resident_chunk_memory.max_image_rows(tables.row_offsets),
            slot_capacity=int(tilt.slot_capacity),
            slot_row_bytes=tilt_slot_row_bytes,
            budget_bytes=tilt_slot_budget_bytes,
        )
    class_slot_block = tilt_slot_block
    row_projection_factor = (1 if tilt is None else int(tilt_slot_block)) * gathered_row_copies
    # The accumulators already exist, so the budget reading counts them; the plan
    # records them so that its pass_bytes is the pass's total need.
    accumulator_bytes = resident_chunk_memory.resident_accumulator_bytes(
        int(np.prod(accumulator_shape)), recon_y_accum_dtype, recon_ctf_accum_dtype, n_slots=int(tables.n_slots)
    )

    # A tilt pass sizes its M-step block for each chunk (resident_chunk_memory.chunk_mstep_block_rows): the pass's
    # block and the plan's agree by this one flag.
    per_chunk_mstep_blocks = tilt is not None

    def plan_chunks(unshifted_operands: bool):
        nonlocal tilt_slot_block, tilt_lone_row_blocks, tilt_lean_lone_rows
        chunk_budget_bytes = resident_chunk_memory.resident_chunk_budget_bytes(reserved_bytes=reserved_operand_bytes)
        image_ladder = engine_tuning.resident_image_capacities or _DEFAULT_IMAGE_CAPACITY_LADDER
        if tilt is not None or not (unshifted_operands and chunk_budget_bytes is not None):
            # The fixed translation-tile budget bounds the translated tiles; an
            # unshifted chunk holds only the Wavg rectangle and its exact
            # positions, which the joint plan below counts against the budget.
            # A tilt pass keeps its resident operands and this start (at S translations).
            image_ladder = resident_chunk_memory.resident_image_capacity_start(
                image_ladder,
                n_fine_trans=plan_translations,
                n_recon_pixels=n_recon_windowed,
                max_tile_bytes=max_translation_tile_bytes_for_pass(
                    device_memory_bytes, override=engine_tuning.translation_tile_max_bytes,
                    has_external_normalization=False,
                ),
                chunk_budget_bytes=chunk_budget_bytes,
            )
        mstep_block_rows = resident_chunk_memory.resolve_mstep_block_rows(
            n_recon_pixels=n_recon_windowed,
            max_block_bytes=max_adjoint_block_bytes_for_pass(
                device_memory_bytes, override=engine_tuning.adjoint_block_max_bytes
            ),
            row_capacity_ladder=row_ladder_start,
            override=engine_tuning.mstep_block_rows,
            per_chunk=per_chunk_mstep_blocks,
        )
        def plan_memory(lone_row_bytes, projection_factor, overflow_projection_factor=None, lone_block_rows=None):
            return resident_chunk_memory.plan_resident_chunk_memory(
                row_capacity_ladder=row_ladder_start,
                image_capacity_ladder=image_ladder,
                mstep_block_rows=mstep_block_rows,
                row_bytes=projection_row_bytes * projection_factor,
                overflow_row_bytes=(
                    None if overflow_projection_factor is None else projection_row_bytes * overflow_projection_factor
                ),
                n_fine_trans=plan_translations,
                n_recon_pixels=n_recon_windowed,
                budget_bytes=chunk_budget_bytes,
                rows_live_during_prepare=True,
                pipelined=tilt is None and _global_chunk_loop_pipelined(stream_projections, chunk_timing=engine_tuning.chunk_timing),
                float32_posterior_buckets=(
                    _FLOAT32_BUCKETED_IMAGE_SUMS
                    and tilt is None
                    and np.dtype(precision_policy.score_real_dtype) == np.dtype(np.float32)
                ),
                fixed_bytes=accumulator_bytes,
                max_image_rows=resident_chunk_memory.max_image_rows(tables.row_offsets),
                lone_row_bytes=lone_row_bytes,
                lone_block_rows=lone_block_rows,
                per_chunk=per_chunk_mstep_blocks,
                **resident_chunk_memory.chunk_translated_tile_pixels(
                    unshifted_operands=unshifted_operands,
                    n_score_pixels=n_windowed if windowed_prepare else n_half_pixels,
                    n_recon_pixels=n_recon_windowed if windowed_prepare else n_half_pixels,
                    n_rect_pixels=n_rect,
                    n_exact_rect_pixels=int(relion_wavg_rectangle.exact_positions.size),
                    normalized_cc=bool(firstiter_cc),
                    masked_scoring=bool(score_with_masked_images),
                ),
            )

        if tilt is None:
            memory_plan = plan_memory(resident_chunk_memory.lone_chunk_row_bytes(int(plan_translations)), row_projection_factor)
        else:
            tilt_slot_block, tilt_lone_row_blocks, tilt_lean_lone_rows = class_slot_block, False, None
            try:
                # A particle past the largest row class is one chunk holding all of its rows, when that fits.
                memory_plan = plan_memory(
                    None, row_projection_factor, int(tilt_overflow_slot_block) * gathered_row_copies
                )
            except ResidentConfigurationUnsupported:
                if firstiter_cc or not resident_chunk_memory.overflow_row_capacity(resident_chunk_memory.max_image_rows(tables.row_offsets), row_ladder_start):
                    raise
                # It does not: the particle runs alone in row blocks of the largest row class
                # (resident_tilts.run_tilt_chunk), which hold rows x slot block x pixels projection values
                # while every row keeps tilt_lone_row_bytes. The slot block is then sized for a block's rows,
                # not for the particle's.
                lone_slot_block = tilt_projection_slot_block(
                    largest_row_class,
                    slot_capacity=int(tilt.slot_capacity),
                    slot_row_bytes=tilt_slot_row_bytes,
                    budget_bytes=chunk_budget_bytes,
                )
                try:
                    memory_plan = plan_memory(
                        tilt_lone_row_bytes(
                            int(n_fine_trans), slot_capacity=int(tilt.slot_capacity), n_accumulators=int(tables.n_slots)
                        ),
                        int(lone_slot_block) * gathered_row_copies,
                    )
                except ResidentConfigurationUnsupported:
                    # Nor does that (relax#118: 294,912 rows x 1,176 translations, 11 GiB of posterior arrays
                    # against a 16 GB card's 6.7 GiB): the particle takes its lean form, at the largest block of
                    # rows down from the largest row class that fits.
                    memory_plan, lone_slot_block, tilt_lean_lone_rows = plan_lean_lone_blocks(
                        plan_memory,
                        lean_row_bytes=tilt_lone_row_bytes(
                            int(n_fine_trans),
                            slot_capacity=int(tilt.slot_capacity),
                            n_accumulators=int(tables.n_slots),
                            lean=True,
                        ),
                        lone_rows=resident_chunk_memory.overflow_row_capacity(
                            resident_chunk_memory.max_image_rows(tables.row_offsets), row_ladder_start
                        ),
                        largest_row_class=largest_row_class,
                        slot_capacity=int(tilt.slot_capacity),
                        slot_row_bytes=tilt_slot_row_bytes,
                        budget_bytes=chunk_budget_bytes,
                        gathered_row_copies=gathered_row_copies,
                    )
                tilt_slot_block, tilt_lone_row_blocks = int(lone_slot_block), True
                logger.info(
                    "Resident pass-2 tilt particles past the largest row class run in row blocks: the largest "
                    "(%d rows) does not fit one chunk; image slot block %d of %d%s",
                    resident_chunk_memory.max_image_rows(tables.row_offsets),
                    tilt_slot_block,
                    int(tilt.slot_capacity),
                    ""
                    if tilt_lean_lone_rows is None
                    else f"; lean form (scores in place, reconstruction-only posterior) in blocks of "
                    f"{tilt_lean_lone_rows} rows",
                )
        row_ladder = memory_plan.row_capacity_ladder
        image_ladder = memory_plan.image_capacity_ladder
        mstep_block_rows = memory_plan.mstep_block_rows
        chunks = plan_pass_chunks(
            tables,
            row_capacity_ladder=row_ladder,
            image_capacity_ladder=image_ladder,
            image_ranges=list(tables.blocks()),
        )
        plan = resident_chunk_memory.ResidentPass2Plan(
            chunks=tuple(chunks),
            row_capacity_ladder=tuple(row_ladder),
            image_capacity_ladder=tuple(image_ladder),
            mstep_block_rows=int(mstep_block_rows),
        )
        table_s = time.time() - table_t0
        row_slots = sum(int(chunk.row_capacity) for chunk in chunks)
        image_slots = sum(int(chunk.image_capacity) for chunk in chunks)
        logger.info(
            "Resident pass-2 plan: %d images, %d candidate rows in %d table blocks -> %d chunks "
            "(row capacities %s, image capacities %s, M-step block rows %d, "
            "row occupancy %.3f of %d slots, image occupancy %.3f of %d slots; "
            "chunk peak %.2f GiB of a %s budget, pass need %.2f GiB with %.2f GiB of accumulators); "
            "setup hypothesis_prep=%.2fs table+plan=%.2fs",
            tables.n_images,
            tables.n_rows,
            tables.n_blocks,
            len(chunks),
            ",".join(str(v) for v in plan.row_capacity_ladder),
            ",".join(str(v) for v in plan.image_capacity_ladder),
            plan.mstep_block_rows,
            tables.n_rows / max(row_slots, 1),
            row_slots,
            tables.n_images / max(image_slots, 1),
            image_slots,
            memory_plan.peak_bytes / float(1024**3),
            resident_chunk_memory.format_budget_gib(memory_plan.budget_bytes),
            memory_plan.pass_bytes / float(1024**3),
            accumulator_bytes / float(1024**3),
            prep_s,
            table_s,
        )

        return memory_plan, row_ladder, image_ladder, mstep_block_rows, chunks, plan

    # ---- preparation arguments --------------------------------------------
    # One keyword set, used by whichever preparation the pass selects: the
    # once-per-half resident preparation below, or the per-chunk call that
    # stays as its oracle. The preparation is per-image pure, so the two return
    # the same rows; the resident form runs it once for the half instead of
    # once per chunk, which is where the chunk loop's launches came from.
    bucket_io_kwargs = dict(
        noise_variance_half=noise_variance_half,
        fine_translations=spa_fine_translations,
        config=config,
        finite_check_mode=engine_tuning.finite_check,
        n_trans=n_fine_trans,
        score_with_masked_images=score_with_masked_images,
        half_spectrum_scoring=half_spectrum_scoring,
        image_corrections=image_corrections,
        scale_corrections=scale_corrections,
        image_pre_shifts=image_pre_shifts,
        use_float64_scoring=use_float64_scoring,
        score_only=False,
        score_mode=relion_firstiter_score_mode,
        window_indices=window_indices,
        recon_window_indices=recon_window_indices,
        translation_phases_half=translation_phases_half,
        relion_score_translation_angles=relion_score_translation_angles,
        return_windowed_shifted=windowed_prepare,
        relion_exact_normalized_cc_operands=relion_exact_fine_normalized_cc,
        relion_exact_bpref_operands=relion_exact_bpref_operands,
        noise_optics_groups=optics_groups_np,
    )
    if nyquist_column_counting != "relion":
        bucket_io_kwargs["nyquist_column_counting"] = nyquist_column_counting
    if image_translations is not None:
        bucket_io_kwargs["image_translations"] = np.asarray(image_translations, dtype=np.float64)
    if firstiter_cc and firstiter_cc_support != "relion":
        # Xi2 over the pixels the normalized CC counts (the score window's weights).
        bucket_io_kwargs["cc_power_weights"] = half_weights_windowed
    # Without the half's resident operands (streamed projections, or operands
    # too large), a chunk prepares its own images' unshifted operands (T16)
    # where they are supported, instead of translated tiles: several times
    # smaller per image, so a chunk takes more images. At K=1 100k/256 hp3 the
    # tiles held 32 images per chunk, 1560 chunks per half.
    chunk_unshifted_supported = resident_chunk_inputs.chunk_unshifted_operands_supported(
        bucket_io_kwargs,
        window_indices=window_indices,
        recon_window_indices=recon_window_indices,
        firstiter_cc=firstiter_cc,
        relion_native_fine_units=relion_native_fine_units,
        relion_exact_bpref_operands=relion_exact_bpref_operands,
        requested=engine_tuning.resident_operands,
    )
    planned_unshifted = bool(reserved_operand_bytes) or chunk_unshifted_supported
    memory_plan, row_ladder, image_ladder, mstep_block_rows, chunks, plan = plan_chunks(planned_unshifted)

    # ---- resident row-aligned tables --------------------------------------
    mstep_grid = (
        fine_grid
        if fine_mstep_rotations_override is None
        else jnp.asarray(fine_mstep_rotations_override, dtype=precision_policy.score_real_dtype)
    )
    coarse_parent_np = np.asarray(fine_rotation_parent_override, dtype=np.int32)
    cached_slot_fine_rot = None
    if projection_slots is not None:
        # Indexed by cache row: the slot's fine rotation and (class, coarse rotation) mass slot.
        slot_rotation = projection_slots.slot_projection % n_fine_rot
        slot_class = projection_slots.slot_projection // n_fine_rot
        mstep_grid = mstep_grid[jnp.asarray(slot_rotation, dtype=jnp.int32)]
        coarse_parent_np = (
            coarse_parent_np[slot_rotation] + slot_class.astype(np.int32) * np.int32(n_coarse_rot)
        ).astype(np.int32)
        cached_slot_fine_rot = jnp.asarray(slot_rotation, dtype=jnp.int32)
    elif n_classes > 1:
        # Indexed by projection id: the M-step rotation is the class's fine
        # rotation, and the rotation-mass slot is (class, coarse rotation).
        mstep_grid = jnp.tile(mstep_grid, (n_classes, 1, 1))
        coarse_parent_np = (
            np.arange(n_classes, dtype=np.int32)[:, None] * np.int32(n_coarse_rot) + coarse_parent_np[None, :]
        ).reshape(-1)
        if not stream_projections:
            cached_slot_fine_rot = jnp.asarray(np.tile(np.arange(n_fine_rot, dtype=np.int32), n_classes))
    coarse_parent_grid = jnp.asarray(coarse_parent_np, dtype=jnp.int32)
    projection_score_cache = (
        None if score_cache is None else score_cache if isinstance(score_cache, tuple) else jnp.asarray(score_cache)
    )
    projection_recon_cache = None if recon_cache is None else jnp.asarray(recon_cache)
    projection_recon_abs2_cache = None if recon_abs2_cache is None else jnp.asarray(recon_abs2_cache)
    fine_translation_parent_device = jnp.asarray(fine_translation_parent, dtype=jnp.int32)

    scale_corrections_np = (
        None
        if scale_corrections is None
        else np.asarray(scale_corrections, dtype=precision_policy.score_real_dtype)
    )

    # ---- statistics accumulators ------------------------------------------
    stats_config = resident_statistics.resolve_statistics_config(
        n_shells=n_shells,
        n_fine_trans=n_fine_trans,
        n_images=n_units,
        n_coarse_rot=n_classes * n_coarse_rot,
        n_scale_groups=n_scale_groups,
        # The norm's unweighted cutoff is the weighted sums' size (no stable windows when it differs).
        current_size=program_current_size if wsum_current_size == current_size else wsum_current_size,
        include_unweighted_high_shell=include_unweighted_norm_high_shell,
        relion_wavg_atomic_scale_aa=relion_wavg_atomic_scale_aa,
        accumulate_scale=accumulate_scale,
        source_faithful_spectrum_norm=resolved_spectrum_norm,
        n_optics_groups=n_optics_groups,
        n_classes=n_classes,
        float32_bucketed_image_sums=_FLOAT32_BUCKETED_IMAGE_SUMS,
        # A local search's compact grid changes size every iteration (resolve_statistics_config).
        rotation_capacity=coarse_rotation_ids is not None,
    )
    stats = resident_statistics.make_resident_statistics(
        stats_config, max_posterior_dtype=precision_policy.score_real_dtype
    )
    image_tables = resident_image_terms.ChunkImageTables(
        shell_indices_half=jnp.asarray(shell_indices_half, dtype=jnp.int32),
        wavg_shell_indices=jnp.asarray(relion_wavg_rectangle.shell_indices, dtype=jnp.int32),
        wavg_scale_pixel_mask=jnp.asarray(scale_pixel_mask_rect_np, dtype=bool),
        translation_sqdist_ang=None,
    )
    window_logical = resident_stable_windows.window_logical_sizes(
        current_size=current_size,
        recon_pixels=logical_recon_pixels,
        rect_pixels=logical_rect_pixels,
        place=resident_chunk_records.PLACE_ON_DEVICE,
        wsum_current_size=None if wsum_current_size == current_size else wsum_current_size,
        # The capacity cube's adjoint clips at RELION's radius: its compact
        # trilinear bound and 3-D radius check read it (recovar backproject_indexed).
        mstep_max_r=None if stable_window_plan is None else resident_stable_windows.runtime_mstep_radius(mstep_max_r),
    )

    max_adjoint_block_bytes = max_adjoint_block_bytes_for_pass(
        device_memory_bytes, override=engine_tuning.adjoint_block_max_bytes
    )
    exact_positions_device = jnp.asarray(relion_wavg_rectangle.exact_positions, dtype=jnp.int32)
    rect_indices_device = jnp.asarray(relion_wavg_rectangle.centered_indices, dtype=jnp.int32)
    noise_variance_for_noise_device = jnp.asarray(noise_variance_for_noise)
    shell_indices_noise_device = jnp.asarray(shell_indices_noise, dtype=jnp.int32)
    recon_pixel_indices_device = jnp.asarray(recon_window_indices, dtype=jnp.int32)

    # ---- T16: per-image operands prepared once for the whole half ----------
    # The per-chunk preparation repeats this work for every chunk an image
    # appears in (image occupancy 0.38-0.66 at the early state) and was 71.5% of
    # the half's CUDA launches. The operands are per-image pure, so one pass
    # over the half produces the same rows; the chunk loop then gathers them.
    resident_operands = None
    warmup = None
    resident_operand_kwargs = dict(
        bucket_io_kwargs=bucket_io_kwargs,
        window_indices=window_indices,
        recon_window_indices=recon_window_indices,
        wavg_rect_indices=relion_wavg_rectangle.centered_indices,
        noise_shell_indices_half=shell_indices_half,
        n_noise_shells=int(n_shells),
        image_shape=image_shape,
        # The operands' powerClass terms sum above the weighted sums' size (ml_optimiser.cpp:6381-6404).
        current_size=wsum_current_size,
        n_fine_trans=int(n_fine_trans),
        use_exact_relion_gaussian=use_exact_relion_gaussian,
        accumulate_noise=accumulate_noise,
        source_faithful_spectrum_norm=resolved_spectrum_norm,
        fine_translation_prior_2d=fine_translation_prior_2d,
        scale_corrections_np=scale_corrections_np,
        group_ids_np=group_ids_np,
        precision_policy=precision_policy,
        optics_groups_np=optics_groups_np,
        relion_native_fine_units=relion_native_fine_units,
        allow_normalized_cc=tilt is not None,
    )
    # The --firstiter_cc iteration scores the compact engine's translated
    # normalized-CC tiles, which only the per-chunk preparation builds; it is
    # one iteration at a small current size.
    # A tilt pass's CC iteration translates each image's resident CC operands per image slot (resident_tilts).
    if engine_tuning.resident_operands and (not firstiter_cc or tilt is not None):
        operand_bytes, operand_peak_bytes = resident_projections.resident_half_operand_sizes(
            n_images=n_images,
            n_windowed=n_windowed,
            n_recon_windowed=n_recon_windowed,
            n_rect=n_rect,
            n_shells=n_shells,
            n_fine_trans=n_fine_trans,
            precision_policy=precision_policy,
            resolved_spectrum_norm=resolved_spectrum_norm,
            premultiplied_ctf=ctf_premultiplied_pass,
            image_shape=image_shape,
        )
        _, _norm_high_shell_dtype = relion_powerclass_noise_dtypes(
            real_dtype=precision_policy.score_real_dtype,
            source_faithful_spectrum_norm=resolved_spectrum_norm,
        )
        # Measured as the streamed projection budget is, after this pass's
        # projection cache exists: what the allocator can still hand out. Same
        # predicate as the reservation before the cache decision.
        available_bytes = current_device_available_bytes()
        budget_bytes = resident_operands_max_bytes(available_bytes, engine_tuning.resident_operand_max_bytes)
        if stream_keeps_chunk_operands:
            logger.info(
                "Resident pass-2 keeps the per-chunk operand preparation: %s, and one half's "
                "resident operands (%.2f GiB) would sit next to that working set",
                "the projections are streamed per chunk"
                if stream_projections
                else "the whole-grid projection cache fits only without them",
                operand_bytes / float(1024**3),
            )
        elif not resident_projections.resident_operands_fit(
            operand_peak_bytes, available_bytes, engine_tuning.resident_operand_max_bytes
        ):
            logger.info(
                "Resident pass-2 keeps the per-chunk operand preparation: one half's resident "
                "operands would take %.2f GiB (%.2f GiB while preparing) against a %.2f GiB budget",
                operand_bytes / float(1024**3),
                operand_peak_bytes / float(1024**3),
                budget_bytes / float(1024**3),
            )
        else:
            # ---- P4-J: compile the chunk programs while the operands prepare -
            # Everything the chunk programs are keyed on is decided by now, and
            # the preparation below is 2.4-5.8s of host-bound device dispatch
            # that leaves the compiler idle. Off unless asked for; a warm-up
            # that describes the wrong program costs its own compile time and
            # changes nothing else, so the two log lines after the block, not an
            # assertion, are what report it.
            warm_pool = CompileAheadPool(engine_tuning.compile_ahead)
            warm_predicted = None
            with warm_pool:
                # The warm-up describes the cached tables; a streamed pass keys
                # its programs on chunk-local tables, so it compiles in the loop.
                if engine_tuning.compile_ahead.enabled and not stream_projections and n_classes == 1:
                    # A warm-up must never fail a run. The pool swallows a
                    # failure on its helper thread; this covers the submission
                    # itself, which runs here on the main thread and reaches
                    # into the plan, the tables and the operand predictor.
                    try:
                        warm_t0 = time.time()
                        presence = resident_half_operand_presence(
                            relion_exact_bpref_operands=bool(
                                bucket_io_kwargs.get("relion_exact_bpref_operands")
                            ),
                            use_exact_relion_gaussian=use_exact_relion_gaussian,
                            accumulate_noise=accumulate_noise,
                            current_size=wsum_current_size,
                        )
                        warm_predicted = resident_half_operand_avals(
                            n_images=int(n_images),
                            n_score_pixels=int(n_windowed),
                            n_recon_pixels=int(n_recon_windowed),
                            n_rect_pixels=int(n_rect),
                            n_noise_shells=int(n_shells),
                            n_fine_trans=int(n_fine_trans),
                            score_complex_dtype=precision_policy.score_complex_dtype,
                            score_real_dtype=precision_policy.score_real_dtype,
                            acc_real_dtype=jnp.float64 if use_float64_scoring else jnp.float32,
                            norm_high_shell_dtype=_norm_high_shell_dtype,
                            has_recon_weight=presence.has_recon_weight,
                            has_direct_ctf_rfloat=presence.has_direct_ctf_rfloat,
                            has_highres_xi2=presence.has_highres_xi2,
                            has_relion_norm_high_shell=presence.has_relion_norm_high_shell,
                            has_optics_groups=optics_groups_np is not None,
                            has_bpref_ctf2_over_nv=ctf_premultiplied_pass,
                        )
                        warmup = resident_chunk_programs.submit_resident_chunk_warmup(
                            warm_pool,
                            # A lone overflow chunk runs its own row-blocked stages.
                            chunks=[chunk for chunk in chunks if not resident_chunk_memory.chunk_runs_alone(chunk, row_ladder)],
                            tables=tables,
                            n_fine_trans=int(n_fine_trans),
                            half_operand_avals=warm_predicted,
                            stage_tables=resident_chunk_programs.make_chunk_stage_tables(
                                projection_score_cache=projection_score_cache,
                                projection_recon_cache=projection_recon_cache,
                                projection_recon_abs2_cache=projection_recon_abs2_cache,
                                mstep_grid=mstep_grid,
                                coarse_parent_grid=coarse_parent_grid,
                                fine_translation_parent_device=fine_translation_parent_device,
                                half_weights=jnp.asarray(half_weights_windowed),
                                translation_angles=jnp.asarray(
                                    relion_score_translation_angles, dtype=jnp.float32
                                ),
                                full_to_compact=relion_score_full_to_compact,
                                noise_variance_for_noise=noise_variance_for_noise_device,
                                shell_indices_noise=shell_indices_noise_device,
                                exact_positions_device=exact_positions_device,
                                recon_pixel_indices=recon_pixel_indices_device,
                                relion_x_half_recon_indices=relion_x_half_recon_indices,
                                image_tables=image_tables,
                                coarse_reuse=coarse_reuse,
                                window_logical=window_logical,
                            ),
                            carry=(Ft_y_total, Ft_ctf_total, stats),
                            translation_angles=jnp.asarray(
                                relion_score_translation_angles, dtype=jnp.float32
                            ),
                            rect_indices=rect_indices_device,
                            exact_positions=exact_positions_device,
                            image_shape=image_shape,
                            spec_kwargs=dict(
                                n_fine_trans=n_fine_trans,
                                n_score_pixels=n_windowed,
                                n_recon_pixels=n_recon_windowed,
                                n_rect=n_rect,
                                mstep_block_rows=mstep_block_rows,
                                adaptive_fraction=adaptive_fraction,
                                current_size=program_current_size,
                                mstep_current_size=program_volume_current_size,
                                mstep_max_r=program_mstep_max_r,
                                image_shape=image_shape,
                                recon_volume_shape=program_recon_volume_shape,
                                max_adjoint_block_bytes=max_adjoint_block_bytes,
                                stats_config=stats_config,
                                use_rfloat_ctf_wavg=presence.has_direct_ctf_rfloat,
                                use_translate_sum_kernel=True,
                                bpref_recon_operand=presence.has_recon_weight,
                                glue_jit=engine_tuning.resident_glue_jit,
                                reuse_coarse_normalization=coarse_reuse is not None,
                                n_slots=int(tables.n_slots),
                                mstep_subtract_ctf_projection=bool(mstep_subtract_ctf_projection),
                                stable_window=stable_window_plan is not None,
                                presum_adjoint=presum_adjoint,
                            ),
                            translation_prior_centers_np=translation_prior_centers_np,
                            fine_translations=fine_translations,
                            voxel_size=experiment_dataset.voxel_size,
                            default_translation_sqdist=image_tables.translation_sqdist_ang,
                        )
                        logger.info(
                            "Resident pass-2 compile-ahead: queued %d capacity classes %s "
                            "for the %s chunk path, host cost %.2fs",
                            len(warmup.classes),
                            ",".join(f"{r}x{b}" for r, b in warmup.classes) or "-",
                            warmup.path,
                            time.time() - warm_t0,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.info(
                            "Resident pass-2 compile-ahead could not be submitted (%s: %s); the chunk loop compiles its own programs",
                            type(exc).__name__,
                            exc,
                        )
                        warm_predicted = None
                        warmup = None
                operands_t0 = time.time()
                # The admission counted the smallest batch's working set; the batch takes what the budget leaves.
                prepare_batch = resident_prepare_batch_size(
                    budget_bytes - operand_peak_bytes + resident_prepare_reserved_bytes(image_shape), image_shape,
                    engine_tuning.prepare_image_batch,
                )
                try:
                    resident_operands = prepare_resident_half_operands(
                        experiment_dataset, np.arange(n_images, dtype=np.int64), image_batch_size=prepare_batch,
                        prefetch_depth=engine_tuning.prefetch_depth, **resident_operand_kwargs,
                    )
                except ResidentOperandsUnsupported as reason:
                    logger.info(
                        "Resident pass-2 keeps the per-chunk operand preparation: %s", reason
                    )
                    resident_operands = None
                    chunk_unshifted_supported = False
                else:
                    if int(resident_operands.n_score_pixels) != int(n_windowed):
                        raise ValueError(
                            "resident score operand pixel count does not match the score window: "
                            f"{resident_operands.n_score_pixels} vs {int(n_windowed)}"
                        )
                    if int(resident_operands.n_recon_pixels) != int(n_recon_windowed):
                        raise ValueError(
                            "resident reconstruction operand pixel count does not match the "
                            f"reconstruction window: {resident_operands.n_recon_pixels} vs "
                            f"{int(n_recon_windowed)}"
                        )
                    logger.info(
                        "Resident pass-2 per-half operand preparation: %.2fs",
                        time.time() - operands_t0,
                    )
            # Leaving the block joined the helper: any compile still running
            # when the preparation finished was one the chunk loop was about to
            # wait for anyway.
            if engine_tuning.compile_ahead.enabled:
                logger.info("Resident pass-2 %s", warm_pool.summary)
                for message in warm_pool.summary.errors:
                    logger.info("Resident pass-2 compile-ahead error: %s", message)
                # The prediction was made before the preparation and is compared
                # after it, so this cannot be satisfied by construction. A
                # mismatch means the warm-up described operands the loop will not
                # pass and its compiles were wasted; the run is unaffected.
                if warm_predicted is None:
                    pass
                elif resident_operands is None:
                    logger.info(
                        "Resident pass-2 compile-ahead predicted operands the half did not "
                        "prepare; its programs go unused"
                    )
                else:
                    difference = describe_resident_operand_mismatch(
                        warm_predicted, resident_operands
                    )
                    logger.info(
                        "Resident pass-2 compile-ahead operand prediction: %s",
                        difference if difference else "matches the prepared operands",
                    )
                    # The three spec booleans are the other thing the warm-up has
                    # to predict from configuration rather than read. They key the
                    # program, so getting one wrong warms a signature the loop
                    # never submits even when every operand aval is right.
                    spec_difference = resident_chunk_programs.describe_chunk_spec_prediction(
                        predicted_rfloat_ctf_wavg=presence.has_direct_ctf_rfloat,
                        predicted_bpref_recon_operand=presence.has_recon_weight,
                        predicted_translate_sum_kernel=True,
                        operands=resident_operands,
                    )
                    logger.info(
                        "Resident pass-2 compile-ahead spec prediction: %s",
                        spec_difference if spec_difference
                        else "matches the prepared operands",
                    )

    chunk_unshifted = resident_operands is None and chunk_unshifted_supported
    if chunk_unshifted:
        logger.info("Resident pass-2 prepares each chunk's unshifted per-image operands")
    elif planned_unshifted and resident_operands is None:
        # The chunks were sized for the unshifted operands; the per-chunk
        # preparation holds the translated tiles, several times their size.
        logger.info("Resident pass-2 plans its chunks again for the per-chunk operand preparation")
        memory_plan, row_ladder, image_ladder, mstep_block_rows, chunks, plan = plan_chunks(False)

    verify_operands = resident_operands is not None and engine_tuning.resident_operands_verify

    # ---- chunk loop --------------------------------------------------------
    # With the warm-up on, collect the (program, spec) keys the loop actually
    # submits. Comparing them against what the helper compiled is the hit rate:
    # a key the loop used and the warm-up did not is a program the loop compiled
    # itself, which is what a mis-predicted spec looks like.
    submitted_keys = set() if warmup is not None else None
    loop_t0 = time.time()
    # Software-pipelined as the local pass is: chunk k+1 is enqueued up to its
    # posterior before chunk k's M-step reads its live row ranges back. A
    # streamed pass projects per chunk because the cache did not fit, so it
    # keeps one chunk's projections alive at a time and is not pipelined.
    if tilt is not None:
        from concurrent.futures import ThreadPoolExecutor
        from functools import partial

        from relax.fine_pass.resident_tilts import (
            TiltMstepCensus,
            prepare_tilt_chunk_host,
            run_tilt_chunk,
            tilt_chunk_tile_budget_bytes,
        )

        # Without the half's resident operands (they do not fit), each chunk prepares the same
        # per-image operands for its own tilt images, a consecutive range of the half's images.
        if resident_operands is None:
            logger.info("Resident pass-2 tilt chunks prepare their own images' resident operands")
        tilt_base_tables = resident_chunk_programs.make_chunk_stage_tables(
            projection_score_cache=None,
            projection_recon_cache=None,
            projection_recon_abs2_cache=None,
            mstep_grid=mstep_grid,
            coarse_parent_grid=coarse_parent_grid,
            fine_translation_parent_device=fine_translation_parent_device,
            half_weights=jnp.asarray(half_weights_windowed),
            translation_angles=None,
            full_to_compact=relion_score_full_to_compact,
            noise_variance_for_noise=noise_variance_for_noise_device,
            shell_indices_noise=shell_indices_noise_device,
            exact_positions_device=exact_positions_device,
            recon_pixel_indices=recon_pixel_indices_device,
            relion_x_half_recon_indices=relion_x_half_recon_indices,
            image_tables=image_tables,
        )
        tilt_spec_kwargs = dict(
            n_score_pixels=int(n_windowed),
            n_recon_pixels=int(n_recon_windowed),
            n_rect=n_rect,
            mstep_block_rows=mstep_block_rows,
            adaptive_fraction=float(adaptive_fraction),
            current_size=current_size,
            mstep_current_size=program_volume_current_size,
            mstep_max_r=program_mstep_max_r,
            image_shape=image_shape,
            recon_volume_shape=program_recon_volume_shape,
            max_adjoint_block_bytes=max_adjoint_block_bytes,
            stats_config=stats_config,
            use_translate_sum_kernel=True,
            glue_jit=engine_tuning.resident_glue_jit,
            n_classes=n_classes,
            # VDAM: each tilt image backprojects its residual (cuda_kernel_backproject3D_SGD per image).
            mstep_subtract_ctf_projection=bool(mstep_subtract_ctf_projection),
            firstiter_cc=bool(firstiter_cc),
        )
        unit_image_offsets = np.asarray(tilt.unit_image_offsets, dtype=np.int64)
        mstep_census = TiltMstepCensus()
        n_lone_chunks = 0
        # The projection bytes the plan counted for its largest overflow chunk (plan_memory's overflow factor).
        overflow_projection_bytes = (
            resident_chunk_memory.overflow_row_capacity(resident_chunk_memory.max_image_rows(tables.row_offsets), row_ladder)
            * int(tilt_overflow_slot_block)
            * tilt_slot_row_bytes
        )
        # The tilt M-step's translated tiles (with the translate-and-sum's table) take what the chunk budget leaves
        # (resident_tilts.tilt_chunk_tile_budget_bytes); the preparation's batch working set runs before the chunk's
        # programs and does not share it. Never less than the device share they had (2%).
        tile_floor_bytes = max_translation_tile_bytes_for_pass(
            device_memory_bytes, override=engine_tuning.translation_tile_max_bytes, has_external_normalization=False
        )
        operand_size_kwargs = (
            None
            if resident_operands is not None
            else dict(
                n_windowed=n_windowed,
                n_recon_windowed=n_recon_windowed,
                n_rect=n_rect,
                n_shells=n_shells,
                n_fine_trans=n_fine_trans,
                precision_policy=precision_policy,
                resolved_spectrum_norm=resolved_spectrum_norm,
                premultiplied_ctf=ctf_premultiplied_pass,
                image_shape=image_shape,
            )
        )
        chunk_tile_budget = partial(
            tilt_chunk_tile_budget_bytes,
            memory_plan=memory_plan,
            floor_bytes=tile_floor_bytes,
            explicit_floor=engine_tuning.translation_tile_max_bytes is not None,
            unit_image_offsets=unit_image_offsets,
            operand_size_kwargs=operand_size_kwargs,
        )
        # The capped budgets size the blocks; the whole leftover only widens blocks the cap cuts small
        # (resident_tilt_mstep.mstep_translation_blocks).
        tile_budgets, wide_tile_budgets = [], []
        for chunk in chunks:
            capped, wide = chunk_tile_budget(chunk)
            tile_budgets.append(capped)
            wide_tile_budgets.append(wide)
        if tile_budgets:
            logger.info(
                "Resident pass-2 tilt M-step translation tiles: %.2f-%.2f GiB per chunk, widening blocks up to "
                "%.2f GiB (chunk budget %s GiB, largest chunk peak %.2f GiB, device share %.2f GiB)",
                min(tile_budgets) / float(1024**3),
                max(tile_budgets) / float(1024**3),
                max(wide_tile_budgets) / float(1024**3),
                "none" if memory_plan.budget_bytes is None else f"{memory_plan.budget_bytes / float(1024**3):.2f}",
                memory_plan.peak_bytes / float(1024**3),
                tile_floor_bytes / float(1024**3),
            )
        # The next chunk's host preparation (prepare_tilt_chunk_host: plan-only NumPy) runs on a host thread while
        # this chunk's programs run; the chunks still run one after another, in order.
        from relax.cuda.preprocess import deferred_relion_preprocess_checks

        host_prep = ThreadPoolExecutor(max_workers=1, thread_name_prefix="relax-tilt-host-prep")
        prepare_next = partial(prepare_tilt_chunk_host, tables=tables, tilt=tilt)
        # The image preprocess kernel's finite check is read at the end of the pass, as the coarse pass reads it
        # (particle_coarse_supports): read at each preparation batch, it waited for the device's queued work inside
        # every chunk's operand preparation. A label names the images the chunk prepares, so a preparation made
        # ahead of its chunk is named by its own images.
        with deferred_relion_preprocess_checks("tilt pass 2") as preprocess_checks:
            try:
                next_prepared = host_prep.submit(prepare_next, chunks[0]) if chunks else None
                for chunk_index, chunk in enumerate(chunks):
                    prepared = next_prepared.result()
                    next_prepared = (
                        host_prep.submit(prepare_next, chunks[chunk_index + 1]) if chunk_index + 1 < len(chunks) else None
                    )
                    # An over-budget particle of a row-blocked plan (plan_chunks) runs in blocks of a row class.
                    runs_alone = resident_chunk_memory.chunk_runs_alone(chunk, row_ladder)
                    chunk_lone_block_rows = (
                        (
                            resident_chunk_memory.lone_block_rows(chunk.row_capacity, row_ladder)
                            if tilt_lean_lone_rows is None
                            else int(tilt_lean_lone_rows)
                        )
                        if tilt_lone_row_blocks and runs_alone
                        else None
                    )
                    n_lone_chunks += int(chunk_lone_block_rows is not None)
                    chunk_slot_block = tilt_slot_block
                    if runs_alone and not tilt_lone_row_blocks:
                        # A particle past the row classes: as many slots as keep its projections within the largest one's.
                        chunk_slot_block = tilt_projection_slot_block(
                            int(chunk.row_capacity),
                            slot_capacity=int(tilt.slot_capacity),
                            slot_row_bytes=tilt_slot_row_bytes,
                            budget_bytes=2 * overflow_projection_bytes,
                        )
                    chunk_operands, operand_image_start = resident_operands, 0
                    if chunk_operands is None:
                        operand_image_start = int(unit_image_offsets[int(chunk.image_start)])
                        operand_image_stop = int(unit_image_offsets[int(chunk.image_start) + int(chunk.n_valid_images)])
                        preprocess_checks.at(f"dataset images {operand_image_start}-{operand_image_stop - 1}")
                        chunk_operands = prepare_resident_half_operands(
                            experiment_dataset,
                            np.arange(operand_image_start, operand_image_stop, dtype=np.int64),
                            image_batch_size=engine_tuning.prepare_image_batch,
                            prefetch_depth=engine_tuning.prefetch_depth,
                            **resident_operand_kwargs,
                        )
                    tilt_spec_kwargs.update(
                        use_rfloat_ctf_wavg=chunk_operands.direct_ctf_rfloat_recon is not None,
                        bpref_recon_operand=chunk_operands.recon_weight is not None,
                    )
                    Ft_y_chunk, Ft_ctf_chunk, stats = run_tilt_chunk(
                        chunk,
                        tables=tables,
                        tilt=tilt,
                        resident_operands=chunk_operands,
                        operand_image_start=operand_image_start,
                        project_rotations=project_fine_rotations,
                        base_tables=tilt_base_tables,
                        n_fine_trans=n_fine_trans,
                        spec_kwargs=tilt_spec_kwargs,
                        stats=stats,
                        Ft_y_total=Ft_y_total,
                        Ft_ctf_total=Ft_ctf_total,
                        n_fine_rot=n_fine_rot,
                        image_shape=image_shape,
                        rect_indices_device=rect_indices_device,
                        exact_positions_device=exact_positions_device,
                        tile_budget_bytes=tile_budgets[chunk_index],
                        wide_tile_budget_bytes=wide_tile_budgets[chunk_index],
                        slot_block=chunk_slot_block,
                        mstep_census=mstep_census,
                        row_partial=row_partial,
                        lone_block_rows=chunk_lone_block_rows,
                        lean_lone=chunk_lone_block_rows is not None and tilt_lean_lone_rows is not None,
                        score_pixel_indices=None if window_indices is None else jnp.asarray(window_indices, dtype=jnp.int32),
                        prepared=prepared,
                    )
                    Ft_y_total, Ft_ctf_total = Ft_y_chunk, Ft_ctf_chunk
            finally:
                host_prep.shutdown(wait=True)
        mstep_census.log()
        if n_lone_chunks:
            logger.info("Resident pass-2 ran %d lone overflow particle(s) in row blocks", n_lone_chunks)
    deferred = _global_chunk_loop_pipelined(stream_projections, chunk_timing=engine_tuning.chunk_timing)
    pending = None
    pending_alone = False
    for chunk in chunks if tilt is None else ():
        alone = resident_chunk_memory.chunk_runs_alone(chunk, row_ladder)
        if pending is not None and (alone or pending_alone):
            Ft_y_total, Ft_ctf_total, stats = pending(Ft_y_total, Ft_ctf_total, stats)
            pending = None
        result = resident_chunk_run.run_resident_chunk(
            chunk,
            chunk_timing=engine_tuning.chunk_timing,
            glue_jit=engine_tuning.resident_glue_jit,
            tables=tables,
            window_logical=window_logical,
            experiment_dataset=experiment_dataset,
            bucket_io_kwargs=bucket_io_kwargs,
            half_weights=jnp.asarray(half_weights_windowed),
            translation_angles=jnp.asarray(relion_score_translation_angles, dtype=jnp.float32),
            full_to_compact=relion_score_full_to_compact,
            n_score_pixels=int(n_windowed),
            fine_translation_prior_2d=fine_translation_prior_2d,
            score_real_dtype=precision_policy.score_real_dtype,
            projection_score_cache=projection_score_cache,
            projection_recon_cache=projection_recon_cache,
            projection_recon_abs2_cache=projection_recon_abs2_cache,
            stream_projection_fn=project_ids if stream_projections else None,
            n_fine_rot=n_fine_rot,
            cache_slot_fine_rot=cached_slot_fine_rot,
            slot_of_projection=None if projection_slots is None else projection_slots.slot_of_projection,
            fine_translation_parent_device=fine_translation_parent_device,
            mstep_grid=mstep_grid,
            coarse_parent_grid=coarse_parent_grid,
            n_fine_trans=n_fine_trans,
            n_recon_windowed=n_recon_windowed,
            n_rect=n_rect,
            mstep_block_rows=mstep_block_rows,
            adaptive_fraction=float(adaptive_fraction),
            windowed_prepare=windowed_prepare,
            window_indices=window_indices,
            recon_window_indices=recon_window_indices,
            relion_x_half_recon_indices=relion_x_half_recon_indices,
            exact_positions_device=exact_positions_device,
            rect_indices_device=rect_indices_device,
            recon_pixel_indices=recon_pixel_indices_device,
            resident_operands=resident_operands,
            chunk_unshifted_operands=chunk_unshifted,
            precision_policy=precision_policy,
            verify_operands=verify_operands and chunk is chunks[0],
            image_shape=image_shape,
            current_size=current_size,
            wsum_current_size=None if wsum_current_size == current_size else wsum_current_size,
            # The physical class with stable windows; None keeps RELION's size.
            program_current_size=None if stable_window_plan is None else program_current_size,
            mstep_current_size=program_volume_current_size,
            mstep_max_r=program_mstep_max_r,
            recon_volume_shape=program_recon_volume_shape,
            max_adjoint_block_bytes=max_adjoint_block_bytes,
            noise_variance_for_noise=noise_variance_for_noise_device,
            shell_indices_noise=shell_indices_noise_device,
            group_ids_np=group_ids_np,
            scale_corrections_np=scale_corrections_np,
            translation_prior_centers_np=translation_prior_centers_np,
            fine_translations=fine_translations,
            voxel_size=experiment_dataset.voxel_size,
            use_exact_relion_gaussian=use_exact_relion_gaussian,
            accumulate_noise=accumulate_noise,
            source_faithful_spectrum_norm=resolved_spectrum_norm,
            stats=stats,
            stats_config=stats_config,
            image_tables=image_tables,
            Ft_y_total=Ft_y_total,
            Ft_ctf_total=Ft_ctf_total,
            submitted_keys=submitted_keys,
            optics_groups_np=optics_groups_np,
            relion_native_fine_units=relion_native_fine_units,
            coarse_reuse=coarse_reuse,
            firstiter_cc=firstiter_cc,
            mstep_subtract_ctf_projection=bool(mstep_subtract_ctf_projection),
            union_score_take=union_score_take,
            union_recon_take=union_recon_take,
            union_native_fft_size=(
                native_fft_size if (union_score_take is not None and relion_native_fine_units) else 0
            ),
            residual_sgd_cache=residual_sgd_cache,
            residual_sgd_take=residual_sgd_take,
            deferred=deferred,
            lone_block_rows=resident_chunk_memory.lone_block_rows(chunk.row_capacity, row_ladder) if alone else None,
            row_partial=row_partial,
        )
        if not deferred:
            Ft_y_total, Ft_ctf_total, stats = result
            continue
        if pending is not None:
            Ft_y_total, Ft_ctf_total, stats = pending(Ft_y_total, Ft_ctf_total, stats)
        pending, pending_alone = result, alone
    if pending is not None:
        Ft_y_total, Ft_ctf_total, stats = pending(Ft_y_total, Ft_ctf_total, stats)
    if presum_adjoint:
        adjoint_kwargs = dict(
            window_indices=relion_x_half_recon_indices,
            use_windowed_adjoint=True,
            image_shape=image_shape,
            volume_shape=program_recon_volume_shape,
            disc_type="linear_interp",
            half_image=True,
            half_volume=True,
            max_r=program_mstep_max_r,
            relion_x_half=True,
            max_block_bytes=int(max_adjoint_block_bytes),
            runtime_max_r=None if stable_window_plan is None else window_logical.mstep_max_r,
        )
        Ft_y_total, Ft_ctf_total = resident_projections.backproject_projection_sums(
            Ft_y_total,
            Ft_ctf_total,
            mstep_grid=mstep_grid,
            n_classes=n_classes,
            n_fine_rot=n_fine_rot,
            volume_size=program_recon_volume_size,
            adjoint_kwargs=adjoint_kwargs,
        )
    loop_s = time.time() - loop_t0
    if warmup is not None:
        used = submitted_keys or set()
        covered = used & warmup.keys
        missed = used - warmup.keys
        logger.info(
            "Resident pass-2 compile-ahead hit rate: %d of %d programs the chunk "
            "loop submitted were already compiled (%d warmed and unused)%s",
            len(covered),
            len(used),
            len(warmup.keys - used),
            "" if not missed
            else "; missed " + ", ".join(sorted(name for name, _ in missed)),
        )

    # ---- finalize (identical to the compact return block) ------------------
    # RELION symmetriseReconstructions, per class, at the physical cube for a
    # stable-window C1 pass (resident_stable_windows.finalize_stable_bpref).
    Ft_y_out, Ft_ctf_out = [], []
    for class_Ft_y, class_Ft_ctf in zip(Ft_y_total, Ft_ctf_total):
        class_Ft_y, class_Ft_ctf = resident_stable_windows.finalize_stable_bpref(
            class_Ft_y,
            class_Ft_ctf,
            program_recon_volume_shape=program_recon_volume_shape,
            recon_volume_shape=recon_volume_shape,
            symmetry_label=symmetry_label,
            engine_tuning=engine_tuning,
            logger=logger,
            label="Resident pass-2",
            keep_physical_bpref=keep_physical_bpref,
        )
        Ft_y_out.append(class_Ft_y)
        Ft_ctf_out.append(class_Ft_ctf)

    finalized = resident_statistics.finalize_statistics(
        stats, config=stats_config, n_images=n_units, n_coarse_rot=n_classes * n_coarse_rot
    )
    noise_stats = finalized.noise_stats()
    logger.info(
        "Resident pass-2: %d images, %d classes, %d chunks, %.2fs chunk loop, %.2fs total",
        n_images,
        n_classes,
        len(chunks),
        loop_s,
        time.time() - overall_t0,
    )
    return ResidentPass2Result(
        Ft_y=tuple(Ft_y_out),
        Ft_ctf=tuple(Ft_ctf_out),
        n_classes=int(tables.n_classes),
        n_slot_groups=int(tables.n_slot_groups),
        finalized=finalized,
        noise_stats=noise_stats,
        fine_translations=fine_translations,
        score_real_dtype=precision_policy.score_real_dtype,
    )


def _class_accumulators(result, class_index: int):
    """Class ``class_index``'s BPref pair: one volume, or ``[groups, ...]`` with several slot groups.

    The grouped form is the exact-local engine's reconstruction-group layout, which
    VDAM's accumulator adapter (``relax.vdam.estep_common.arrays_to_accumulators``) reads.
    """

    k, n_classes = int(class_index), int(result.n_classes)
    if int(result.n_slot_groups) == 1:
        return result.Ft_y[k], result.Ft_ctf[k]
    slots = [k + n_classes * group for group in range(int(result.n_slot_groups))]
    return _stack_slot_pair(tuple(result.Ft_y[a] for a in slots), tuple(result.Ft_ctf[a] for a in slots))


@jax.jit
def _stack_slot_pair(Ft_y, Ft_ctf):
    # One program per shape; the two eager stacks compiled six.
    return jnp.stack(Ft_y, axis=0), jnp.stack(Ft_ctf, axis=0)


class ResidentPass2Result(NamedTuple):
    """What :func:`_resident_pass2` hands its K=1 and K-class entries."""

    Ft_y: tuple  # per accumulator slot (class + K * group), public x-half layout
    Ft_ctf: tuple
    n_classes: int
    n_slot_groups: int
    finalized: resident_statistics.FinalizedStatistics
    noise_stats: object  # one total over classes (ml_optimiser.cpp:10470, :11010)
    fine_translations: np.ndarray  # score dtype
    score_real_dtype: object


def compute_pass2_stats_resident(
    experiment_dataset,
    volume,
    noise_variance,
    translations,
    significant_sample_indices,
    nside_level,
    disc_type,
    *,
    engine_tuning: relax.runtime.engine_tuning.EngineTuning,
    scoring_overrides,
    oversampling_order,
    current_size,
    reconstruction_current_size=None,
    wsum_current_size=None,
    translation_step,
    rotation_log_prior,
    score_with_masked_images,
    return_stats,
    translation_log_prior,
    accumulate_noise,
    half_spectrum_scoring,
    projection_padding_factor,
    projection_mask_current_image_disk=False,
    reconstruction_padding_factor,
    image_corrections,
    scale_corrections,
    image_pre_shifts,
    use_float64_scoring,
    translation_prior_centers=None,
    do_gridding_correction=False,
    square_window=False,
    window_at_box=False,
    random_perturbation,
    group_ids=None,
    scale_correction_group_count=None,
    scale_correction_data_vs_prior=None,
    normalization_log_z=None,
    relion_f32_normalization_sum_weight=None,
    relion_coarse_hard_assignment=None,
    relion_coarse_max_posterior=None,
    normalization_other_score_log_z=None,
    return_score_log_z=False,
    return_score_log_z_only=False,
    fine_source_eulers_override=None,
    return_source_eulers=False,
    fine_rotations_override=None,
    fine_mstep_rotations_override=None,
    fine_rotation_parent_override=None,
    fine_translations_override=None,
    fine_translation_parent_override=None,
    relion_x_half_mstep=False,
    mstep_subtract_ctf_projection=False,
    relion_fine_mstep_prune=False,
    relion_firstiter_score_mode="gaussian",
    relion_firstiter_winner_take_all=False,
    relion_exact_fine_gaussian=True,
    relion_f32_fine_posterior=False,
    relion_exact_fine_normalized_cc=False,
    relion_projector_half=None,
    relion_projector_texture=None,
    relion_projector_r_max=None,
    adaptive_fraction=0.999,
    include_unweighted_norm_high_shell: bool = True,
    source_faithful_spectrum_norm: bool = False,
    symmetry_label: str = "C1",
    keep_physical_bpref: bool = False,
    relion_translation_angle_scale: float = 1.0,
    optics_group_ids=None,
    reconstruction_volume_current_size=None,
    reconstruction_image_radius=None,
    reconstruction_group_ids=None,
    reconstruction_group_count=None,
    full_grid_engine=None,
    nyquist_column_counting: str = "relion",
    firstiter_cc_support: str = "relion",
    image_translations=None,
):
    """Device-resident K=1 sparse pass 2, relax's one pass-2 engine.

    A one-class :func:`_resident_pass2`. See the module docstring for what is
    layout-equal to the deleted compact engine and what is a deliberate
    reduction-order change.
    """

    # Every parameter, forwarded by name.
    del window_at_box  # _resident_pass2 keeps RELION's window at every size, the box included
    result = _resident_pass2(**locals())
    return _k1_pass2_output(
        result,
        fine_rotations_override=fine_rotations_override,
        fine_source_eulers_override=fine_source_eulers_override,
        return_stats=return_stats,
        return_score_log_z=return_score_log_z,
        return_source_eulers=return_source_eulers,
    )


def compute_tilt_pass2_stats_resident(*, tilt, coarse_rotation_ids=None, unit_rotation_log_prior=None, **options):
    """:func:`compute_pass2_stats_resident` for subtomogram particles over their tilt images (S4.2).

    ``tilt`` is the :class:`relax.fine_pass.resident_tilts.TiltPassInputs`; ``coarse_rotation_ids``
    and ``unit_rotation_log_prior`` are a local search's compact coarse grid and per-particle priors
    (:func:`_resident_pass2`). ``options`` are the K=1 driver's keyword arguments; the per-unit outputs
    are the particles'.
    """

    result = _resident_pass2(
        tilt=tilt,
        coarse_rotation_ids=coarse_rotation_ids,
        unit_rotation_log_prior=unit_rotation_log_prior,
        **options,
    )
    return _k1_pass2_output(
        result,
        fine_rotations_override=options["fine_rotations_override"],
        fine_source_eulers_override=options.get("fine_source_eulers_override"),
        return_stats=options["return_stats"],
        return_score_log_z=options.get("return_score_log_z", False),
        return_source_eulers=options.get("return_source_eulers", False),
    )


def _k1_pass2_output(
    result, *, fine_rotations_override, fine_source_eulers_override, return_stats, return_score_log_z, return_source_eulers
):
    """The compact engine's K=1 output from a one-class resident result."""

    finalized = result.finalized
    hard_assignment = np.asarray(finalized.hard_assignment, dtype=np.int32)
    best_fine_rotation_indices = np.asarray(finalized.best_fine_rotation_indices, dtype=np.int64)
    best_rotations = np.asarray(fine_rotations_override, dtype=result.score_real_dtype)[
        best_fine_rotation_indices
    ]
    best_translations = result.fine_translations[hard_assignment % result.fine_translations.shape[0]]
    best_eulers = None
    if return_source_eulers and fine_source_eulers_override is not None:
        best_eulers = np.asarray(fine_source_eulers_override, dtype=np.float64)[
            best_fine_rotation_indices
        ]
    relion_stats = None
    if return_stats:
        relion_stats = make_relion_stats(
            log_evidence_per_image=finalized.log_evidence_per_image,
            best_log_score_per_image=finalized.best_log_score_per_image,
            max_posterior_per_image=finalized.max_posterior_per_image,
            rotation_posterior_sums=finalized.rotation_posterior_sums,
        )
    Ft_y_class, Ft_ctf_class = _class_accumulators(result, 0)
    return SparsePass2Output(
        Ft_y_class,
        Ft_ctf_class,
        hard_assignment,
        best_rotations,
        best_translations,
        best_fine_rotation_indices,
        relion_stats=relion_stats,
        score_log_z=(
            finalized.score_log_z_per_image if (return_stats and return_score_log_z) else None
        ),
        noise_stats=result.noise_stats,
        source_eulers=best_eulers if return_source_eulers else None,
    )


class ResidentKClassPass2Output(NamedTuple):
    """A K-class resident pass, in the class-segmented result's field names.

    The same names as the exact-local engine's class-segmented output, so one
    adapter (``k_class_sparse_pass2.class_segmented_em_result``) turns either into the
    K-class result. Class axes come first.
    """

    Ft_y: tuple  # per class, public x-half layout
    Ft_ctf: tuple
    class_log_evidence_per_image: np.ndarray  # float64 [K, N], absolute, log pdf_class included
    class_best_log_score_per_image: np.ndarray  # float64 [K, N]
    per_class_hard_assignments: np.ndarray  # int64 [K, N], fine rotation * T + t; -1 without a candidate
    stats: object  # joint RelionStats: log-Z, best, Pmax over classes and poses
    class_rotation_posterior_sums: np.ndarray  # float64 [K, n_coarse_rot]
    class_reconstruction_posterior_sums: np.ndarray  # float64 [K], pruned M-step mass
    noise_stats: object  # one total over classes
    per_class_best_pose_rotations: tuple
    per_class_best_pose_translations: tuple
    per_class_best_pose_rotation_ids: tuple
    per_class_best_pose_eulers_deg: tuple | None
    profile: dict | None = None
    uncast_log_evidence_per_image: np.ndarray | None = None
    # Zero oversampling with the coarse state kept: each image's class, the coarse winner's. None otherwise.
    coarse_class_assignments: np.ndarray | None = None


def compute_k_class_pass2_stats_resident(
    experiment_dataset,
    volumes,
    noise_variance,
    translations,
    significant_sample_indices_by_class,
    nside_level,
    disc_type,
    *,
    rotation_log_priors_by_class,
    **options,
) -> ResidentKClassPass2Output:
    """RELION's Class3D fine pass on the resident engine, every class in one sweep.

    ``volumes`` (and ``relion_projector_half``, when given) stack the K class
    references; ``rotation_log_priors_by_class`` fold each class's
    ``log pdf_class`` into its rotation prior. ``options`` are the K=1 driver's
    keyword arguments. Each image's posterior segment spans all classes, so the
    minimum, the normalization and the significance are RELION's joint ones
    over classes and poses (ml_optimiser.cpp:8411, :9225, :9602-9660); each
    class backprojects into its own BPref (:10826).
    """

    n_classes = len(significant_sample_indices_by_class)
    options.pop("window_at_box", None)  # _resident_pass2 keeps RELION's window at every size
    result = _resident_pass2(
        experiment_dataset,
        volumes,
        noise_variance,
        translations,
        None,
        nside_level,
        disc_type,
        rotation_log_prior=None,
        classes=ResidentClassInputs(
            significant_sample_indices=tuple(significant_sample_indices_by_class),
            rotation_log_priors=tuple(rotation_log_priors_by_class),
        ),
        **options,
    )
    finalized = result.finalized
    per_class = finalized.classes
    fine_rotations = np.asarray(options["fine_rotations_override"], dtype=result.score_real_dtype)
    source_eulers = options.get("fine_source_eulers_override")
    rotation_ids = per_class.best_fine_rotation_indices
    translation_ids = per_class.best_translation_indices
    coarse_classes = options.get("relion_coarse_class_assignment")
    if coarse_classes is not None:
        # Zero oversampling keeps the coarse winner (acc_ml_optimiser_impl.h:1984-1988): its class is the
        # image's, and that class's pose is the winner's one fine child, not the best fine sample.
        coarse_classes = np.asarray(coarse_classes, dtype=np.int64)
        images = np.arange(coarse_classes.size)
        rotation_ids, translation_ids = np.array(rotation_ids), np.array(translation_ids)
        rotation_ids[coarse_classes, images] = finalized.best_fine_rotation_indices
        translation_ids[coarse_classes, images] = finalized.best_translation_indices
    has_pose = rotation_ids >= 0
    safe_rotation = np.where(has_pose, rotation_ids, 0)
    safe_translation = np.where(has_pose, translation_ids, 0)
    n_fine_trans = int(result.fine_translations.shape[0])
    class_accumulators = [_class_accumulators(result, k) for k in range(n_classes)]
    return ResidentKClassPass2Output(
        Ft_y=tuple(pair[0] for pair in class_accumulators),
        Ft_ctf=tuple(pair[1] for pair in class_accumulators),
        class_log_evidence_per_image=per_class.log_evidence,
        class_best_log_score_per_image=per_class.best_log_score,
        per_class_hard_assignments=np.where(
            has_pose, rotation_ids * n_fine_trans + translation_ids, -1
        ).astype(np.int64),
        stats=make_relion_stats(
            log_evidence_per_image=finalized.log_evidence_per_image,
            best_log_score_per_image=finalized.best_log_score_per_image,
            max_posterior_per_image=finalized.max_posterior_per_image,
            rotation_posterior_sums=np.sum(finalized.rotation_posterior_sums, axis=0),
        ),
        class_rotation_posterior_sums=finalized.rotation_posterior_sums,
        class_reconstruction_posterior_sums=per_class.posterior_sums,
        noise_stats=result.noise_stats,
        per_class_best_pose_rotations=tuple(fine_rotations[safe_rotation[k]] for k in range(n_classes)),
        per_class_best_pose_translations=tuple(
            result.fine_translations[safe_translation[k]] for k in range(n_classes)
        ),
        per_class_best_pose_rotation_ids=tuple(safe_rotation[k] for k in range(n_classes)),
        per_class_best_pose_eulers_deg=(
            None
            if source_eulers is None
            else tuple(np.asarray(source_eulers, dtype=np.float64)[safe_rotation[k]] for k in range(n_classes))
        ),
        coarse_class_assignments=None if coarse_classes is None else coarse_classes.astype(np.int32),
    )


def _global_chunk_loop_pipelined(stream_projections: bool, *, chunk_timing: bool) -> bool:
    """Whether the global chunk loop overlaps chunk k+1's front with chunk k's M-step.

    The chunk plan counts the second chunk's arrays (:func:`resident_chunk_bytes`)
    exactly when the loop runs this way. ``chunk_timing`` is ``EngineTuning.chunk_timing``.
    """

    return not chunk_timing and not stream_projections
