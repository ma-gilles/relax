"""Device-resident K=1 sparse pass-2 driver (T9b integration).

This module wires the five stage modules of
``em_device_resident_pass2_design_20260918.md`` into one driver with the
signature and return type of
the compact engine's ``compute_pass2_stats_sparse_bucketed`` (deleted):

* T5 :mod:`recovar.em.sparse_pass2.resident_candidates` -- the flat image-CSR
  candidate table and its fixed-capacity chunks;
* T6 :mod:`recovar.em.sparse_pass2.resident_scoring` -- the resident per-image
  scoring operands and the one-program-per-capacity-class scoring stage;
* T7 ``recovar.cuda_backproject.sparse_pass2_segmented_*`` -- the segmented
  RELION float32 fine posterior over those CSR segments;
* T8 ``recovar.cuda_backproject.relion_wavg_*flat_rows*`` -- the flat-row RELION
  Wavg triplet and its atomic accumulation;
* T9a :mod:`recovar.em.sparse_pass2.resident_statistics` -- the device float64
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
:func:`_accumulate_chunk_image_terms` (T9a's arithmetic, with the row-pixel
reductions arriving as chunk partials) rather than by T9a's single fused
program, which assumes one call per chunk with the whole pixel axis resident.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace as dataclass_replace
from functools import lru_cache, partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from recovar.reconstruction import noise as noise_utils

from relax.fine_pass import (
    policy,
    resident_block_programs,
    resident_chunk_memory,
    resident_chunk_records,
    resident_configuration,
    resident_image_terms,
    resident_projections,
    resident_stable_windows,
)
from relax.fine_pass.adjoint import _accumulate_adjoint_block_chunked
from relax.fine_pass.bucket_arrays import _prepare_per_image_pass2_inputs
from relax.fine_pass.bucket_io import (
    _prepare_bucket_io,
    _relion_cuda_score_translation_angles_if_available,
    premultiplied_bpref_weights,
)
from relax.fine_pass.policy import (
    _RELION_WAVG_ATOMIC_SCALE_AA_ENV,
    ResidentConfigurationUnsupported,
    _projection_cache_enabled_for_pass,
    _relion_wavg_direct_modes,
)
from relax.fine_pass.posterior import (
    _relion_fine_parent_execution_order_enabled,
)
from relax.fine_pass.projection_blocks import (
    _compute_sparse_pass2_windowed_projections_block,
    _projection_kwargs_for_relion_score_window,
    project_rows_by_class,
)
from relax.fine_pass.resident_candidates import (
    _BLOCK_UNITS,
    CandidateTableBlocks,
    chunk_segment_offsets,
    map_over_classes,
    materialize_chunk,
    merge_class_tables,
    plan_pass_chunks,
    table_block_starts,
)
from relax.fine_pass.resident_operands import (
    ResidentOperandsUnsupported,
    describe_resident_operand_mismatch,
    gather_resident_chunk_operands,
    prepare_resident_half_operands,
    require_unshifted_operand_support,
    resident_half_operand_avals,
    resident_half_operand_presence,
    resident_operands_max_bytes,
    resident_prepare_batch_size,
    resident_prepare_reserved_bytes,
)
from relax.fine_pass.resident_scoring import (
    cache_dtype,
    cache_rows,
    score_resident_chunk,
)
from relax.fine_pass.resident_significance import (
    build_resident_candidate_tables_from_csr,
    csr_candidate_rows_per_image,
    fine_rotation_children,
    resident_candidate_tables,
    resident_significance_csr,
)
from relax.fine_pass.resident_statistics import (
    FinalizedStatistics,
    finalize_statistics,
    make_resident_statistics,
    resident_image_capacity,
    resolve_statistics_config,
)
from relax.fine_pass.scoring import (
    _relion_cuda_fine_full_to_compact_lookup,
    _relion_native_fine_units,
    _relion_native_fine_units_enabled,
    _relion_native_score_corr_img,
    _relion_powerclass_noise_terms,
    relion_powerclass_noise_dtypes,
)
from relax.fine_pass.wavg import (
    _make_relion_wavg_rectangle,
    _make_stable_relion_wavg_rectangle,
    _relion_cuda_translate_wavg_norm_images,
    image_power_shells,
)
from relax.fine_pass.window import (
    _pass2_half_weights,
    _pass2_window_setup,
    _sparse_pass2_window_setup,
)
from relax.fourier.half_spectrum import (
    make_relion_noise_shell_indices_half,
    mask_relion_noise_shell_indices_to_current_window,
)
from relax.fourier.preprocessing import half_translation_phase_table
from relax.io.batch_fetch import fetch_indexed_batch
from relax.numerics.deterministic_reduce import deterministic_reductions_enabled
from relax.projection.adjoint import ReferenceSphereClip, mstep_adjoint_max_r
from relax.projection.projection import relion_kernel_zero_rows
from relax.projection.projection import (
    relion_scale_correction_pixel_mask as _relion_scale_correction_pixel_mask,
)
from relax.reconstruction.half_volume_mstep import (
    RowPartial,
    allocate_row_partial,
    crop_public_full_volume,
    crop_relion_x_half_accumulator,
    finalize_half_volume_bpref,
    half_volume_accumulator_shape,
    physical_bpref_finalize_applies,
    relion_backprojector_volume_shape,
    relion_x_half_accumulators_to_public_layout,
    relion_x_half_mstep_accumulator_dtypes,
)
from relax.relion import ctf
from relax.relion.optics_aberrations import dataset_magnification_is_anisotropic, dataset_needs_exact_ctf
from relax.relion.optics_groups import noise_rows, pixel_rows
from relax.relion.scale_groups import prepare_scale_correction_groups
from relax.runtime.compile_ahead import (
    CompileAheadPool,
    resolve_compile_ahead_config,
)
from relax.runtime.env_flags import parse_env_capacity_ladder, parse_env_flag
from relax.runtime.memory_budget import (
    _device_free_memory_bytes,
    _jax_allocator_free_memory_bytes,
    _jax_allocator_pool_free_bytes,
    _max_adjoint_block_bytes_for_pass,
    _max_translation_tile_bytes_for_pass,
    _projection_cache_build_max_rotations_per_call,
    _projection_cache_fits_budget,
    _projection_cache_max_bytes_for_pass,
    _projection_cache_transient_bytes,
    device_available_bytes,
)
from relax.sampling.translation_prior import (
    translation_prior_centers_for_images,
    translation_sqdist_angstrom,
    validate_translation_prior_centers,
)
from relax.types import SparsePass2Output, make_noise_stats, make_relion_stats

logger = logging.getLogger(__name__)

_ROW_CAPACITY_LADDER_ENV = "RELAX_SPARSE_PASS2_RESIDENT_ROW_CAPACITIES"
# The resident pass sums its float32 posterior and norm rows into their images with
# the scatter-add; the bucketed segment_sum (and its planned scratch) is the dense
# GEMM coarse engine's.
_FLOAT32_BUCKETED_IMAGE_SUMS = False
_IMAGE_CAPACITY_LADDER_ENV = "RELAX_SPARSE_PASS2_RESIDENT_IMAGE_CAPACITIES"
# Relative band the racing shell-binning scatter is allowed in the verification
# arm: 12x the measured same-call spread of 5.8e-8, still far inside one
# float32 ulp of the accumulated shell power.
_RACING_SCATTER_RELATIVE_BAND = 7e-7
_SOFT_POSTERIOR_BLOCK_BPREF_PROTOTYPE_ENV = "RELAX_EM_PROTOTYPE_SOFT_POSTERIOR_BLOCK_BPREF"

# Row capacities are multiples of the M-step block so every chunk decomposes
# into whole blocks; image capacities follow the design's ladder. The design's
# three classes, restored after the denser five-class ladder was measured and
# bought nothing: at hp3 the matched pairs put the two ladders inside the
# control's own drift (loop 22.6 versus 22.1 s per half) and at the early state
# they are indistinguishable (resident warm 67.6 / 60.4 versus 67.0 / 61.5 s,
# occupancy 0.95-0.97 either way, jobs 14143902 and 14143904), while the dense
# ladder costs 84 extra traced programs. Every extra class is one more program
# per capacity-class stage.
# The occupancy of each plan is still logged, so a future change has its number.
# 524288 is for early low-SNR states, where each image keeps thousands of rows:
# Class3D K4 100k/256 snr05 iteration 2 ran 8722 chunks of 131072 rows, 2028 of
# 524288, chunk loop 619 -> 523 s and iteration 3 453 -> 374 s (job 14580683).
# The cached-path gather budget (_cached_row_capacity_ladder) and the chunk
# memory plan drop it wherever a chunk that size does not fit.
_DEFAULT_ROW_CAPACITY_LADDER = (8192, 32768, 131072, 524288)
_DEFAULT_IMAGE_CAPACITY_LADDER = (32, 128, 512)

__all__ = [
    "ResidentClassInputs",
    "ResidentKClassPass2Output",
    "compute_k_class_pass2_stats_resident",
    "compute_pass2_stats_resident",
]


# ---------------------------------------------------------------------------
# Production-configuration gate
# ---------------------------------------------------------------------------


def _resident_wavg_arithmetic(
    *,
    accumulate_noise,
    scale_groups_available,
    preserve_bpref_particle_order,
    source_faithful_spectrum_norm,
):
    """Resolve the pass's Wavg arithmetic: RELION's atomic triplet with the direct low-shell residual.

    Returns ``(spectrum_norm, exact_bpref_operands, direct_noise_default,
    atomic_scale_aa)``. The resident statistics stage implements RELION's atomic Wavg triplet
    only, for fresh passes and subset or focused replays alike (user decision 2026-09-26: the
    replays' earlier unordered, non-atomic arithmetic was the compact engine's).
    ``preserve_bpref_particle_order`` no longer changes it.
    """

    del preserve_bpref_particle_order
    # A pass that preserves RELION's BPref order (every fresh K=1 pass) runs RELION's
    # powerClass spectrum with its exact BPref operands; there is no other variant.
    spectrum_norm = exact_bpref_operands = bool(source_faithful_spectrum_norm)
    direct_noise_default = True
    atomic_scale_aa = bool(
        accumulate_noise
        and scale_groups_available
        and parse_env_flag(_RELION_WAVG_ATOMIC_SCALE_AA_ENV, default=direct_noise_default)
    )
    return spectrum_norm, exact_bpref_operands, direct_noise_default, atomic_scale_aa


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
# Planning
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Device programs with a pixel axis (one per (block rows, pixel count) pair)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Per-chunk image-level statistics (every term without a row-pixel axis)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _pad_batch_to_capacity(values, capacity: int):
    """Repeat a host batch's first row up to ``capacity`` rows.

    Every per-chunk device program is keyed on its operand shapes, so a batch
    whose length is the chunk's *valid* image count traces a new program for
    every distinct occupancy. Padding the batch on the host, before anything
    is traced, gives one program per image-capacity class instead. The padded
    rows carry a duplicate image's real data and are zeroed after preparation
    by :func:`_zero_padded_images`, which is itself capacity-shaped.
    """

    values = np.asarray(values)
    n = int(values.shape[0])
    if n == capacity:
        return values
    if n > capacity or n == 0:
        raise ValueError(f"cannot pad a batch of {n} rows to capacity {capacity}")
    pad = np.repeat(values[:1], capacity - n, axis=0)
    return np.concatenate([values, pad], axis=0)


# Host images are fetched in sub-batches below glibc's largest dynamic mmap
# threshold (32 MiB), so each fetch reuses heap memory, and are copied into one
# reused capacity-sized buffer. A 512-image fetch at box 800 (1.3 GB) otherwise
# faulted in fresh pages twice per chunk, for the gather and for the padding:
# about 95 s per half of host time in the EMPIAR-10202 pass-1 probe
# (py-spy, bigbox 14747301), against 5 s for the exact engine's per-image reads.
_FETCH_SUB_BATCH_BYTES = 24 << 20
_CAPACITY_BATCH_BUFFERS: dict = {}


def fetch_capacity_batch(experiment_dataset, image_indices, capacity: int):
    """One chunk's host images padded to ``capacity`` and uploaded, with its CTF rows and indices.

    Returns ``(images, ctf_params, fetched_indices, padded_fetched_indices)``:
    the device images ``[capacity, ...]`` in fetched order, padded with the first
    fetched image as :func:`_pad_batch_to_capacity` pads, the padded CTF rows,
    and the fetched indices without and with that padding. The values are the
    unpadded fetch's; only where the host copies live changes.
    """

    image_indices = np.asarray(image_indices)
    n = int(image_indices.shape[0])
    if n > capacity or n == 0:
        raise ValueError(f"cannot pad a batch of {n} rows to capacity {capacity}")
    buffer = None
    ctf_parts, fetched_parts = [], []
    start, step = 0, 1
    while start < n:
        images, ctf_params, fetched = fetch_indexed_batch(experiment_dataset, image_indices[start : start + step])
        images = np.asarray(images)
        if buffer is None:
            key = ((capacity,) + tuple(images.shape[1:]), images.dtype.str)
            buffer = _CAPACITY_BATCH_BUFFERS.get(key)
            if buffer is None:
                _CAPACITY_BATCH_BUFFERS.clear()
                buffer = _CAPACITY_BATCH_BUFFERS[key] = np.empty(key[0], dtype=images.dtype)
            step = max(1, _FETCH_SUB_BATCH_BYTES // max(int(images[0].nbytes), 1))
        buffer[start : start + images.shape[0]] = images
        ctf_parts.append(np.asarray(ctf_params))
        fetched_parts.append(np.asarray(fetched))
        start += int(images.shape[0])
    buffer[n:] = buffer[0]
    # The upload is waited for: the buffer is refilled by the next chunk.
    device_images = jax.device_put(buffer)
    device_images.block_until_ready()
    fetched_indices = np.concatenate(fetched_parts)
    return (
        device_images,
        _pad_batch_to_capacity(np.concatenate(ctf_parts), capacity),
        fetched_indices,
        _pad_batch_to_capacity(fetched_indices, capacity),
    )


def _reorder_permutation(fetched_indices, requested_indices, capacity: int) -> np.ndarray:
    """Host permutation from the fetched batch order to the table order.

    The dataset may return a batch in its own order. The compact engine
    reorders its bucket arrays to follow the fetch; the resident driver keeps
    the table's (RELION particle) order and permutes the operands, which is a
    pure gather and changes no arithmetic. Padded slots point at fetched row
    0; :func:`_zero_padded_images` removes whatever they gathered.
    """

    fetched = np.asarray(fetched_indices).reshape(-1)
    requested = np.asarray(requested_indices).reshape(-1)
    n = int(requested.shape[0])
    position_of = {int(index): position for position, index in enumerate(requested.tolist())}
    if len(position_of) != n:
        raise ValueError("a chunk must not request the same image twice")
    order = np.zeros(capacity, dtype=np.int32)
    seen = np.zeros(n, dtype=bool)
    for slot, dataset_index in enumerate(fetched[:n].tolist()):
        position = position_of.get(int(dataset_index))
        if position is None:
            raise ValueError(f"the dataset returned image {dataset_index}, which was not requested")
        order[position] = slot
        seen[position] = True
    if not bool(seen.all()):
        raise ValueError("the dataset did not return every requested image")
    return order


class _ChunkOperandRowInputs(NamedTuple):
    """The per-chunk operands that are permuted into row order and zero-padded.

    Each optional field is ``None`` on the paths that do not produce it;
    ``None`` is a pytree structure, so those paths key their own program
    rather than carry a dead operand.
    """

    score_input: jax.Array
    corr_img_score: jax.Array
    highres_xi2_half: jax.Array | None
    shifted_recon: jax.Array
    shifted_noise: jax.Array
    ctf2_over_nv_recon: jax.Array
    direct_ctf_rfloat_recon: jax.Array | None
    processed_score_half_for_noise: jax.Array
    relion_norm_high_shell: jax.Array | None
    raw_translated_wavg_rectangle: jax.Array
    bpref_ctf2_over_nv_recon: jax.Array | None = None


@partial(jax.jit, static_argnames=("image_capacity", "n_fine_trans"))
def _chunk_operand_rows(
    arrays: _ChunkOperandRowInputs,
    permutation: jax.Array,
    valid_images: jax.Array,
    exact_positions: jax.Array,
    *,
    image_capacity: int,
    n_fine_trans: int,
) -> tuple:
    """Permute a chunk's operands into row order and zero the padded slots.

    Same statements, same order, same dtypes as the loose dispatch: two
    reshapes, ten permutation gathers, ten ``where`` masks and one rectangle
    gather. None of it is arithmetic, so no value can move; what leaves the
    host is the dispatch count. Eagerly, ``values[permutation]`` is five
    primitives rather than one, because JAX normalizes a fancy index
    (``add``, ``broadcast_in_dim``, ``select_n``) before every gather, and the
    resident local pass runs this once per chunk.
    """

    def take(values):
        return None if values is None else _zero_padded_images(
            values[permutation], valid_images
        )

    raw_translated_wavg_rectangle = take(arrays.raw_translated_wavg_rectangle)
    return (
        take(arrays.score_input),
        take(arrays.corr_img_score),
        take(arrays.highres_xi2_half),
        take(arrays.shifted_recon.reshape(image_capacity, n_fine_trans, -1)),
        take(arrays.shifted_noise.reshape(image_capacity, n_fine_trans, -1)),
        take(arrays.ctf2_over_nv_recon),
        take(arrays.direct_ctf_rfloat_recon),
        take(arrays.processed_score_half_for_noise),
        take(arrays.relion_norm_high_shell),
        raw_translated_wavg_rectangle,
        raw_translated_wavg_rectangle[:, :, exact_positions],
        take(arrays.bpref_ctf2_over_nv_recon),
    )


def _zero_padded_images(values, valid_images):
    """Zero the padded image slots of a capacity-shaped operand.

    ``valid_images`` is a capacity-shaped bool, so this traces one program per
    (capacity, trailing shape) pair regardless of how many slots are valid.
    """

    values = jnp.asarray(values)
    mask = jnp.asarray(valid_images, dtype=bool).reshape((-1,) + (1,) * (values.ndim - 1))
    return jnp.where(mask, values, jnp.zeros((), dtype=values.dtype))


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
    padding = resident_image_capacity(tables.n_images) - tables.n_images
    return resident_chunk_records.CoarseNormalizationReuse(
        sum_weight=jnp.asarray(np.pad(sum_weight, (0, padding)), dtype=jnp.float64),
        max_posterior=jnp.asarray(np.pad(max_posterior, (0, padding)), dtype=max_posterior_dtype),
        winner_cell=jnp.asarray(np.pad(np.asarray(winner_cell), (0, padding), constant_values=-1), dtype=jnp.int64),
    )


def _class_candidate_tables(
    significant_sample_indices,
    rotation_log_prior,
    *,
    n_images,
    n_coarse_rot,
    n_coarse_trans,
    nside_level,
    oversampling_order,
    n_fine_trans,
    fine_translation_parent,
    random_perturbation,
    fine_source_eulers_override,
    fine_rotations_override,
    fine_mstep_rotations_override,
    fine_rotation_parent_override,
    use_relion_f32_fine_posterior,
    dtype,
    symmetry_label,
    coarse_rotation_ids=None,
    per_image_rotation_log_prior=None,
):
    """One class's per-image hypotheses and candidate table (T5), exactly the K=1 build.

    ``coarse_rotation_ids`` and ``per_image_rotation_log_prior`` are the compact coarse grid and
    the per-unit priors of a subtomogram local search (``_prepare_per_image_pass2_inputs``).
    Returns ``(tables, hypothesis_prep_seconds, table_seconds)``.
    """

    prep_t0 = time.time()
    relion_parent_execution_order = _relion_fine_parent_execution_order_enabled(
        use_relion_f32_fine_posterior=use_relion_f32_fine_posterior,
    )
    significance_csr = resident_significance_csr(
        significant_sample_indices,
        n_images=n_images,
        n_coarse_rot=n_coarse_rot,
        n_coarse_trans=n_coarse_trans,
    )
    per_image_inputs = None if significance_csr is not None else _prepare_per_image_pass2_inputs(
        significant_sample_indices,
        n_coarse_rot=n_coarse_rot,
        n_coarse_trans=n_coarse_trans,
        nside_level=nside_level,
        oversampling_order=oversampling_order,
        n_fine_trans=n_fine_trans,
        fine_translation_parent=fine_translation_parent,
        rotation_log_prior=rotation_log_prior,
        random_perturbation=random_perturbation,
        fine_source_eulers_override=fine_source_eulers_override,
        fine_rotations_override=fine_rotations_override,
        fine_mstep_rotations_override=fine_mstep_rotations_override,
        fine_rotation_parent_override=fine_rotation_parent_override,
        relion_parent_execution_order=relion_parent_execution_order,
        dtype=dtype,
        symmetry_label=symmetry_label,
        coarse_rotation_ids=coarse_rotation_ids,
        per_image_rotation_log_prior=per_image_rotation_log_prior,
    )
    if significance_csr is not None and (coarse_rotation_ids is not None or per_image_rotation_log_prior is not None):
        raise ValueError("a compact coarse grid or per-image priors take the host candidate-table path")
    prep_s = time.time() - prep_t0

    table_t0 = time.time()
    tables = resident_candidate_tables(
        significance_csr,
        per_image_inputs,
        n_coarse_trans=n_coarse_trans,
        n_fine_trans=n_fine_trans,
        fine_translation_parent=fine_translation_parent,
        nside_level=nside_level,
        oversampling_order=oversampling_order,
        rotation_log_prior=rotation_log_prior,
        random_perturbation=random_perturbation,
        fine_rotation_parent_override=fine_rotation_parent_override,
        relion_parent_execution_order=relion_parent_execution_order,
        dtype=dtype,
        symmetry_label=symmetry_label,
    )
    if int(tables.n_images) != int(n_images):
        raise ValueError(
            f"candidate table covers {tables.n_images} images but the dataset has {n_images}"
        )
    return tables, prep_s, time.time() - table_t0


def _candidate_table_blocks(
    class_supports,
    class_rotation_priors,
    *,
    whole: bool,
    reconstruction_group_ids,
    reconstruction_group_count,
    **table_kwargs,
):
    """The pass's candidate tables over every class, as :class:`CandidateTableBlocks`.

    With every class's device-compacted CSR the tables are built one image
    block of at most ``_BLOCK_UNITS`` rows plus significant ids at a time, when
    the chunk loop first reaches the block; only the per-image row counts are
    computed here. The
    rows, their order and their fields are the whole-pass build's, since each
    image's rows depend on that image alone. ``whole`` (the zero-oversampling
    coarse reuse reads every image's table) and a host-path support build the
    one-block tables as before. Returns ``(blocks, hypothesis_prep_s, table_s)``.
    """

    n_images = int(table_kwargs["n_images"])
    csrs = [
        resident_significance_csr(
            support,
            n_images=n_images,
            n_coarse_rot=table_kwargs["n_coarse_rot"],
            n_coarse_trans=table_kwargs["n_coarse_trans"],
        )
        for support in class_supports
    ]

    def with_groups(tables, start, stop):
        group_ids = None if reconstruction_group_ids is None else np.asarray(reconstruction_group_ids)[start:stop]
        return _with_reconstruction_groups(tables, group_ids, reconstruction_group_count, n_images=stop - start)

    def merged(tables_by_class):
        return tables_by_class[0] if len(tables_by_class) == 1 else merge_class_tables(tables_by_class)

    # A subtomogram local search's compact coarse grid and per-unit priors take the host build.
    compact_local = (
        table_kwargs.get("coarse_rotation_ids") is not None
        or table_kwargs.get("per_image_rotation_log_prior") is not None
    )
    if whole or compact_local or any(csr is None for csr in csrs):
        class_builds = map_over_classes(
            lambda item: _class_candidate_tables(item[0], item[1], **table_kwargs),
            zip(class_supports, class_rotation_priors),
        )
        table_t0 = time.time()
        tables = with_groups(merged([built for built, _, _ in class_builds]), 0, n_images)
        prep_s = sum(class_prep_s for _, class_prep_s, _ in class_builds)
        table_s = sum(class_table_s for _, _, class_table_s in class_builds) + time.time() - table_t0
        return CandidateTableBlocks.whole(tables), prep_s, table_s

    table_t0 = time.time()
    children = fine_rotation_children(
        n_coarse_rot=table_kwargs["n_coarse_rot"],
        nside_level=table_kwargs["nside_level"],
        oversampling_order=table_kwargs["oversampling_order"],
        random_perturbation=table_kwargs["random_perturbation"],
        fine_rotation_parent_override=table_kwargs["fine_rotation_parent_override"],
        symmetry_label=table_kwargs["symmetry_label"],
    )
    rows_per_image = np.sum(
        map_over_classes(lambda csr: csr_candidate_rows_per_image(csr, children[0]), csrs), axis=0
    )
    row_offsets = np.zeros(n_images + 1, dtype=np.int64)
    row_offsets[1:] = np.cumsum(rows_per_image)
    # A block is counted in rows plus the classes' significant ids its builders read (_BLOCK_UNITS).
    unit_offsets = row_offsets.copy()
    unit_offsets[1:] += np.cumsum(np.sum([csr.counts() for csr in csrs], axis=0, dtype=np.int64))
    csr_kwargs = dict(
        nside_level=table_kwargs["nside_level"],
        oversampling_order=table_kwargs["oversampling_order"],
        n_fine_trans=table_kwargs["n_fine_trans"],
        fine_translation_parent=table_kwargs["fine_translation_parent"],
        random_perturbation=table_kwargs["random_perturbation"],
        fine_rotation_parent_override=table_kwargs["fine_rotation_parent_override"],
        relion_parent_execution_order=_relion_fine_parent_execution_order_enabled(
            use_relion_f32_fine_posterior=table_kwargs["use_relion_f32_fine_posterior"],
        ),
        dtype=table_kwargs["dtype"],
        symmetry_label=table_kwargs["symmetry_label"],
        children=children,
    )

    def build_block(start, stop):
        return with_groups(
            merged(
                map_over_classes(
                    lambda item: build_resident_candidate_tables_from_csr(
                        item[0].image_block(start, stop), rotation_log_prior=item[1], **csr_kwargs
                    ),
                    zip(csrs, class_rotation_priors),
                )
            ),
            start,
            stop,
        )

    blocks = CandidateTableBlocks(
        n_images=n_images,
        n_fine_trans=int(table_kwargs["n_fine_trans"]),
        n_coarse_trans=int(table_kwargs["n_coarse_trans"]),
        n_classes=len(csrs),
        n_slot_groups=1 if reconstruction_group_count is None else int(reconstruction_group_count),
        row_offsets=row_offsets,
        block_starts=table_block_starts(unit_offsets, _BLOCK_UNITS),
        build_block=build_block,
    )
    return blocks, 0.0, time.time() - table_t0


class ResidentClassInputs(NamedTuple):
    """The class axis of a K-class pass (RELION Class3D), one entry per class.

    ``rotation_log_priors`` are each class's coarse rotation log prior with its
    ``log pdf_class`` folded in, as the compact engine takes them
    (k_class.py ``_class_rotation_prior``).
    """

    significant_sample_indices: tuple
    rotation_log_priors: tuple


def _runtime_mstep_radius(mstep_max_r) -> float:
    """The adjoint's runtime radius in image Fourier pixels: ``max_r``, or a sphere clip's image radius."""

    return float(mstep_max_r.image_radius) if isinstance(mstep_max_r, ReferenceSphereClip) else float(mstep_max_r)


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
    sigma_min = float(np.linalg.svd(np.asarray(rotations, dtype=np.float64), compute_uv=False).min())
    reachable = (sigma_min * label) ** 2 < float(r_max) ** 2 + 1.0
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
    normalization_score_mode=None,
    return_score_log_z=False,
    return_score_log_z_only=False,
    rotation_block_size_for_quantization=5000,
    fine_source_eulers_override=None,
    return_source_eulers=False,
    fine_rotations_override=None,
    fine_mstep_rotations_override=None,
    fine_rotation_parent_override=None,
    fine_translations_override=None,
    fine_translation_parent_override=None,
    relion_half_volume_mstep=False,
    relion_x_half_mstep=False,
    mstep_subtract_ctf_projection=False,
    relion_fine_mstep_prune=False,
    relion_firstiter_score_mode="gaussian",
    relion_firstiter_winner_take_all=False,
    relion_exact_fine_gaussian=True,
    relion_fine_diff2_fused_ffi=False,
    relion_f32_fine_posterior=False,
    relion_exact_fine_normalized_cc=False,
    relion_projector_half=None,
    relion_projector_texture=None,
    relion_projector_r_max=None,
    adaptive_fraction=0.999,
    bpref_device_signature_active: bool = False,
    bpref_class_index: int = 0,
    include_unweighted_norm_high_shell: bool = True,
    preserve_bpref_particle_order: bool = False,
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
    """The device-resident sparse pass 2 over one or K classes; returns ``_ResidentPass2Result``.

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
    here: the scoring weights (``_pass2_half_weights``), the image the noise statistics read
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
        _fine_translation_prior_2d,
        _pass2_projection_budget,
        _pass2_relion_flags,
    )
    from relax.sampling.grids import rotation_grid_size
    from relax.sampling.oversampled_grids import get_oversampled_translation_grid, infer_translation_step
    from relax.sampling.symmetry import canonicalize_rotational_symmetry

    _require_gpu_pass2()
    overall_t0 = time.time()
    (
        use_exact_relion_gaussian,
        _use_relion_fine_diff2_fused_ffi,
        use_relion_f32_fine_posterior,
    ) = _pass2_relion_flags(
        relion_exact_fine_gaussian=relion_exact_fine_gaussian,
        relion_firstiter_score_mode=relion_firstiter_score_mode,
        relion_fine_diff2_fused_ffi=relion_fine_diff2_fused_ffi,
        relion_f32_fine_posterior=relion_f32_fine_posterior,
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
        # particle carries its own orientation prior over its support's coarse rotations.
        resident_configuration.require(tilt is not None and classes is None, "a compact coarse grid is the subtomogram local search's")
        resident_configuration.require(rotation_log_prior is None, "a local search's rotation priors are the particles' own")
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
        ) = _pass2_window_setup(
            image_shape,
            current_size=current_size,
            reconstruction_current_size=reconstruction_current_size,
            half_spectrum_scoring=half_spectrum_scoring,
            square_window=square_window,
            relion_firstiter_score_mode=relion_firstiter_score_mode,
            use_exact_relion_gaussian=use_exact_relion_gaussian,
            use_float64_scoring=use_float64_scoring,
            # RELION's window at every size, including the box (a shape class reaches its
            # box before the reference does): the resident driver never scores a full half.
            window_at_box=True,
            reference_sphere_clip=reconstruction_image_radius is not None or anisotropic_magnification,
        )
    except NotImplementedError as exc:
        # A window the resident scorer does not implement is a configuration gap,
        # reported before any device work like the checks below.
        raise ResidentConfigurationUnsupported(str(exc)) from exc

    scale_groups_available = group_ids is not None
    # RELION runs the Wavg triplet whether or not it corrects scales: without
    # --scale every particle's scale is 1 and only the XA/AA sums are skipped
    # (acc_ml_optimiser_impl.h:4367-4372, 4907, 4980). The triplet therefore runs
    # without groups too (scale 1, group -1, so no scale sums accumulate); the
    # scale statistics themselves stay tied to real groups (``accumulate_scale``).
    wavg_triplet_available = bool(scale_groups_available or accumulate_noise)
    (
        resolved_spectrum_norm,
        relion_exact_bpref_operands,
        direct_noise_default,
        relion_wavg_atomic_scale_aa,
    ) = _resident_wavg_arithmetic(
        accumulate_noise=accumulate_noise,
        scale_groups_available=wavg_triplet_available,
        preserve_bpref_particle_order=preserve_bpref_particle_order,
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
    relion_native_fine_units = _relion_native_fine_units_enabled(
        fresh_k1_guard=bool(source_faithful_spectrum_norm),
        use_exact_relion_gaussian=use_exact_relion_gaussian,
        use_float64_scoring=use_float64_scoring,
        has_ctf_rfloat=relion_exact_bpref_operands,
    ) and not (firstiter_cc and tilt is not None)  # a tilt CC pass translates its score image (resident_tilts)
    native_fft_size = int(np.prod(image_shape))
    relion_wavg_atomic_direct_noise, relion_wavg_atomic_direct_norm = _relion_wavg_direct_modes(
        accumulate_noise=bool(accumulate_noise),
        scale_groups_available=wavg_triplet_available,
        scale_aa_enabled=bool(relion_wavg_atomic_scale_aa),
        direct_noise_only_default=direct_noise_default,
    )

    soft_posterior_block_bpref = parse_env_flag(
        _SOFT_POSTERIOR_BLOCK_BPREF_PROTOTYPE_ENV, default=True
    )
    projection_cache_enabled = _projection_cache_enabled_for_pass(
        fine_rotations_override=fine_rotations_override,
        dump_pass2_operands=False,
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
        preserve_bpref_particle_order=preserve_bpref_particle_order,
        soft_posterior_block_bpref=soft_posterior_block_bpref,
        fine_rotations_override=fine_rotations_override,
        fine_rotation_parent_override=fine_rotation_parent_override,
        use_window=budget_window_spec.use_window,
        projection_cache_available=projection_cache_enabled,
        relion_wavg_atomic_scale_aa=relion_wavg_atomic_scale_aa,
        relion_wavg_atomic_direct_noise=relion_wavg_atomic_direct_noise,
        relion_wavg_atomic_direct_norm=relion_wavg_atomic_direct_norm,
        relion_projector_texture=relion_projector_texture,
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
    volume_current_size = (
        mstep_current_size
        if reconstruction_volume_current_size is None
        else int(reconstruction_volume_current_size)
    )
    recon_volume_shape = relion_backprojector_volume_shape(
        volume_shape,
        reconstruction_padding_factor,
        current_size=volume_current_size,
    )
    recon_accum_shape = half_volume_accumulator_shape(recon_volume_shape)
    recon_volume_size = int(np.prod(recon_accum_shape))
    mstep_max_r = mstep_adjoint_max_r(
        volume_current_size, reconstruction_image_radius, reconstruction_padding_factor,
        anisotropic_magnification=anisotropic_magnification,
    )
    recon_y_accum_dtype, recon_ctf_accum_dtype = relion_x_half_mstep_accumulator_dtypes(
        experiment_dataset.dtype,
        use_relion_x_half_mstep=True,
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
            # 97.7% equal, 14384091) and twice the iteration time.
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
    fine_translation_prior_2d = _fine_translation_prior_2d(
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
            relion_parent_execution_order=_relion_fine_parent_execution_order_enabled(
                use_relion_f32_fine_posterior=use_relion_f32_fine_posterior,
            ),
            current_size=int(current_size), mstep_current_size=int(mstep_current_size),
            n_half=n_half, window_spec_kwargs=window_spec_kwargs,
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
            relion_wavg_atomic_direct_noise=relion_wavg_atomic_direct_noise,
            relion_wavg_atomic_scale_aa=relion_wavg_atomic_scale_aa,
            accumulate_scale=scale_groups_available,
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
    # (_candidate_table_blocks).
    tables, prep_s, table_s = _candidate_table_blocks(
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
    stable_window_plan = None if tilt is not None or wsum_current_size != current_size else resident_stable_windows.resident_stable_window_plan(
        image_shape,
        current_size=current_size,
        mstep_current_size=mstep_current_size,
        n_half=n_half,
        square_window=square_window,
        window_spec_kwargs=window_spec_kwargs,
        firstiter_cc=firstiter_cc,
    )
    stable_window_spec = None
    # The spec's current size: the physical class with stable windows, RELION's otherwise.
    program_current_size = int(current_size)
    if stable_window_plan is not None:
        # Capacity changes storage, not the projection or reconstruction
        # cutoffs: the projector keeps the logical radius and crop, so the
        # tail's projections never enter a logical pixel.
        stable_window_spec = dataclass_replace(
            stable_window_plan.packed_physical_spec(),
            max_r=stable_window_plan.logical_spec.max_r,
            projection_max_r=stable_window_plan.logical_spec.projection_max_r,
            image_current_size=stable_window_plan.logical_spec.image_current_size,
        )
        budget_window_spec = stable_window_spec
        program_current_size = int(stable_window_plan.physical_current_size)
        logger.info(
            "Resident pass-2 stable windows: logical current_size %d -> physical class %d "
            "(score pixels %d -> %d, recon pixels %d -> %d, Wavg rectangle %d -> %d)",
            stable_window_plan.logical_current_size,
            stable_window_plan.physical_current_size,
            stable_window_plan.logical_score_pixels,
            stable_window_plan.physical_score_pixels,
            stable_window_plan.logical_reconstruction_pixels,
            stable_window_plan.physical_reconstruction_pixels,
            stable_window_plan.logical_rectangle_pixels,
            stable_window_plan.physical_rectangle_pixels,
        )
    window_setup = _sparse_pass2_window_setup(
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

    half_weights, half_weights_windowed = _pass2_half_weights(
        image_shape,
        window_spec,
        half_spectrum_scoring=half_spectrum_scoring,
        relion_firstiter_score_mode=relion_firstiter_score_mode,
        use_float64_scoring=use_float64_scoring,
        nyquist_column_counting=nyquist_column_counting,
        firstiter_cc_support=firstiter_cc_support,
        current_size=current_size,
    )
    if stable_window_plan is None:
        relion_score_full_to_compact = jnp.asarray(
            _relion_cuda_fine_full_to_compact_lookup(image_shape, current_size, window_indices_np),
            dtype=jnp.int32,
        )
    else:
        # The score tail gets no weight, and the fine scorer's lookup covers the
        # logical rectangle only (the runtime kernel never reads past it); its
        # -1 pad keeps the buffer at the physical class's size.
        half_weights_windowed = _times_logical_mask(
            half_weights_windowed, jnp.int32(int(stable_window_plan.logical_score_pixels))
        )
        logical_lookup = _relion_cuda_fine_full_to_compact_lookup(
            image_shape,
            current_size,
            window_indices_np[: int(stable_window_plan.logical_score_pixels)],
        )
        relion_score_full_to_compact = jnp.asarray(
            np.pad(
                logical_lookup,
                (0, int(stable_window_plan.physical_rectangle_pixels) - logical_lookup.size),
                constant_values=-1,
            ),
            dtype=jnp.int32,
        )
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(
        noise_variance, image_shape
    ).squeeze()
    n_optics_groups = 1 if noise_variance_half.ndim == 1 else int(noise_variance_half.shape[0])
    optics_groups_np = None
    if n_optics_groups > 1:
        if optics_group_ids is None:
            raise ValueError("a per-optics-group noise table needs optics_group_ids")
        optics_groups_np = np.asarray(optics_group_ids, dtype=np.int32).reshape(-1)
        if optics_groups_np.shape != (n_images,) or np.any(optics_groups_np < 0) or np.any(
            optics_groups_np >= n_optics_groups
        ):
            raise ValueError(
                f"optics_group_ids must give each of {n_images} images a row of the "
                f"{n_optics_groups}-group noise table"
            )
    # Tilt images (S4.2) carry 3D translations whose phases are per image (tilt.image_angles); the SPA
    # operand preparation, which never sees a trial shift of a tilt image, gets 2D zeros of the same count.
    spa_fine_translations_source = (
        fine_translations_source if tilt is None else np.zeros((n_fine_trans, 2), dtype=np.float64)
    )
    spa_fine_translations = (
        fine_translations if tilt is None else np.zeros((n_fine_trans, 2), dtype=fine_translations.dtype)
    )
    relion_score_translation_angles = _relion_cuda_score_translation_angles_if_available(
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
    # The crop mask sees RELION's logical window, not the physical class; with a weighted-sum size
    # above the scoring size it is that size's crop rectangle (the Wavg rectangle's).
    noise_crop_indices = (
        window_indices
        if stable_window_plan is None
        else window_indices_np[: int(stable_window_plan.logical_score_pixels)]
    )
    if wsum_current_size != current_size:
        noise_crop_indices = _make_relion_wavg_rectangle(
            image_shape,
            wsum_current_size,
            recon_window_indices,
            reconstruction_current_size=mstep_current_size,
        ).centered_indices
    shell_indices_half = mask_relion_noise_shell_indices_to_current_window(
        make_relion_noise_shell_indices_half(image_shape),
        image_shape,
        wsum_current_size,
        noise_crop_indices,
    )
    shell_indices_noise = window_spec.recon_values(shell_indices_half)
    noise_variance_for_noise = window_spec.recon_values(noise_variance_half)
    if stable_window_plan is None:
        relion_wavg_rectangle = _make_relion_wavg_rectangle(
            image_shape,
            wsum_current_size,
            recon_window_indices,
            reconstruction_current_size=mstep_current_size,
        )
        logical_rect_pixels = int(relion_wavg_rectangle.centered_indices.size)
        logical_recon_pixels = int(n_recon_windowed)
    else:
        # The recon tail holds real pixels above RELION's cutoff. They take the
        # shell sentinel and zero noise variance, so every residual, power and
        # scale term they could enter is exactly zero; the M-step masks their
        # sums (_resident_mstep_block) and the Wavg kernels stop at the logical
        # rectangle, whose layout is RELION's byte for byte.
        shell_indices_noise, noise_variance_for_noise = _mask_recon_tail(
            shell_indices_noise,
            noise_variance_for_noise,
            jnp.int32(int(stable_window_plan.logical_reconstruction_pixels)),
            shell_sentinel=int(image_shape[0] // 2 + 1),
        )
        relion_wavg_rectangle = _make_stable_relion_wavg_rectangle(image_shape, stable_window_plan)
        logical_rect_pixels = int(stable_window_plan.logical_rectangle_pixels)
        logical_recon_pixels = int(stable_window_plan.logical_reconstruction_pixels)
    n_rect = int(relion_wavg_rectangle.centered_indices.size)
    # The BPref accumulators and the adjoint are keyed on the physical class
    # too: the pass accumulates into the physical current-size cube and crops
    # it to RELION's logical cube before finalizing (crop_relion_x_half_accumulator,
    # as the local engine does). The logical pixels land on the same Fourier
    # voxels of either cube, and the M-step gives the tail no weight.
    program_volume_current_size = int(volume_current_size)
    program_recon_volume_shape = tuple(int(v) for v in recon_volume_shape)
    program_mstep_max_r = mstep_max_r
    if stable_window_plan is not None:
        program_volume_current_size = (
            int(stable_window_plan.physical_reconstruction_current_size)
            if reconstruction_volume_current_size is None
            else resident_stable_windows.stable_reference_volume_class(int(volume_shape[0]), int(volume_current_size))
        )
        program_recon_volume_shape = tuple(
            int(v)
            for v in relion_backprojector_volume_shape(
                volume_shape, reconstruction_padding_factor, current_size=program_volume_current_size
            )
        )
        program_mstep_max_r = mstep_adjoint_max_r(
            program_volume_current_size,
            resident_stable_windows.reference_image_radius_at(reconstruction_image_radius, volume_current_size, program_volume_current_size),
            reconstruction_padding_factor,
            anisotropic_magnification=anisotropic_magnification,
        )
    program_recon_volume_size = int(np.prod(half_volume_accumulator_shape(program_recon_volume_shape)))
    # One x-half BPref pair per accumulator slot: RELION's BPref[iclass], and for
    # VDAM's pseudo-halfsets BPref[iclass + half * nr_classes]. They exist before
    # the pass reads free device memory (projection cache, resident operand and
    # chunk budgets), so every budget sees them; allocated after the chunk
    # budget they were not counted, up to 23 GiB at EMPIAR-10202's full box.
    # A K-class pass whose sums fit carries per-projection row sums instead
    # (_projection_sums_bytes); the volumes are then built after the chunk loop.
    # K=1 keeps the per-row backprojection: it measured no gain (K1 50k, job
    # 14585996: 1095 -> 1092 s), and its cold-start parity cases carry the
    # reassociation into the trajectory (fast tier k1_os1_coldstart_standalone
    # half-1 FSC-AUC 1.000000 -> 0.998679, bisect 14587006).
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
    free_bytes = resident_projections.allocator_available_bytes()
    # A tilt chunk backprojects each (image, row) with its own matrix (resident_tilts), never per projection.
    presum_adjoint = (
        n_classes > 1
        and tilt is None
        and free_bytes is not None
        and sums_bytes <= resident_projections.PRESUM_ADJOINT_FREE_FRACTION * float(free_bytes)
        and not resident_projections.projection_sums_displace_resident_operands(
            sums_bytes,
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
    zero_accumulators = _zero_block_partials(
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
        else allocate_row_partial(program_recon_volume_size, recon_y_accum_dtype, recon_ctf_accum_dtype)
    )
    logger.info(
        "Resident pass-2 M-step adjoint: %s (per-projection sums %.2f GiB, %s GiB available to the allocator)",
        "per-projection sums, one backprojection per pass" if presum_adjoint else "per-row backprojection",
        sums_bytes / float(1024**3),
        "unknown" if free_bytes is None else f"{float(free_bytes) / float(1024**3):.2f}",
    )
    # RELION masks each class's scale sums with its own data_vs_prior_class[iclass] > 3
    # (acc_ml_optimiser_impl.h:4908); one shell vector serves every class.
    scale_dvp_by_class = [scale_correction_data_vs_prior] * n_classes
    if scale_correction_data_vs_prior is not None and n_classes > 1:
        scale_dvp_array = np.asarray(scale_correction_data_vs_prior)
        if scale_dvp_array.ndim == 2:
            if scale_dvp_array.shape[0] != n_classes:
                raise ValueError(
                    "scale_correction_data_vs_prior must be one shell vector or have "
                    f"shape ({n_classes}, n_shells), got {scale_dvp_array.shape}"
                )
            scale_dvp_by_class = [scale_dvp_array[k] for k in range(n_classes)]
    scale_pixel_mask_rect_np = np.zeros((n_classes, n_rect), dtype=bool)
    for class_index, class_dvp in enumerate(scale_dvp_by_class):
        scale_pixel_mask_rect_np[class_index, relion_wavg_rectangle.exact_positions] = np.asarray(
            _relion_scale_correction_pixel_mask(class_dvp, shell_indices_noise, n_shells=n_shells),
            dtype=bool,
        )
    if n_classes == 1:
        scale_pixel_mask_rect_np = scale_pixel_mask_rect_np[0]
    group_ids_np, n_scale_groups = prepare_scale_correction_groups(
        group_ids, scale_correction_group_count, n_images=n_images,
    )

    # ---- projection cache (same admission and build as the compact engine) -
    # K>1 caches every class's fine grid, class k at k * n_fine_rot; when the
    # significant coarse parents' children are far fewer, only they are cached
    # (_significant_projection_slots), class k at k * capacity.
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
    ) = _pass2_projection_budget(
        jnp.asarray(class_means_for_proj[0]).dtype,
        precision_policy,
        n_half=n_half,
        use_relion_projector=use_relion_projector,
        budget_window_spec=budget_window_spec,
        device_memory_bytes=device_memory_bytes,
        include_abs2=False,
    )
    transient_projection_bytes = _projection_cache_transient_bytes(
        n_projections,
        n_windowed,
        projection_complex_dtype=precision_policy.score_complex_dtype,
        include_abs2=False,
    ) + _projection_cache_transient_bytes(
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
        cache_projection_bytes = _projection_cache_transient_bytes(
            n_projections,
            int(union_indices_np.size),
            projection_complex_dtype=precision_policy.score_complex_dtype,
            include_abs2=False,
        )
    max_projection_cache_bytes = _projection_cache_max_bytes_for_pass(device_memory_bytes)
    projection_kwargs = _projection_kwargs_for_relion_score_window(
        window_spec.projection_kwargs(return_abs2=False),
        use_relion_projector=use_relion_projector,
        current_size=current_size,
    )
    projection_kwargs["mask_current_image_disk"] = bool(projection_mask_current_image_disk)
    # Stable windows also project at the physical class: a center-padded capacity
    # slab into the physical crop, with RELION's logical model and image radii as
    # runtime values (_capacity_projection). The projection programs then keep one
    # shape per class instead of compiling at every logical current size.
    projection_halves = class_projector_halves
    projection_r_max = relion_projector_r_max
    capacity_projection_kwargs = {}
    capacity_logical_output_size = None
    if stable_window_plan is not None and resident_stable_windows.capacity_projection_applies(
        use_relion_projector=use_relion_projector,
        class_projector_halves=class_projector_halves,
        relion_projector_r_max=relion_projector_r_max,
        projection_padding_factor=projection_padding_factor,
        projection_kwargs=projection_kwargs,
        physical_current_size=int(stable_window_plan.physical_current_size),
    ):
        physical_size = int(stable_window_plan.physical_current_size)
        projection_halves = [
            resident_stable_windows.center_pad_relion_projector_half(
                half,
                logical_r_max=int(relion_projector_r_max),
                physical_size=physical_size,
                padding_factor=int(projection_padding_factor),
            )
            for half in class_projector_halves
        ]
        # The capacity route takes the model radius at runtime; its static radius is the sentinel 0.
        projection_r_max = 0
        capacity_projection_kwargs = {
            "relion_projector_runtime_r_max": jnp.asarray(int(relion_projector_r_max), dtype=jnp.int32),
            "relion_projector_image_size": jnp.asarray(int(projection_kwargs["projector_output_size"]), dtype=jnp.int32),
        }
        capacity_logical_output_size = int(projection_kwargs["projector_output_size"])
        projection_kwargs["projector_output_size"] = physical_size
    residual_sgd_rows = (
        _sgd_residual_rows(
            recon_window_indices,
            image_shape=image_shape,
            projector_output_size=(
                projection_kwargs.get("projector_output_size")
                if capacity_logical_output_size is None
                else capacity_logical_output_size
            ),
            r_max=relion_projector_r_max,
            rotations=np.asarray(fine_rotations_override, dtype=np.float64),
        )
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

        score, recon, recon_abs2 = _compute_sparse_pass2_windowed_projections_block(
            class_means_for_proj[class_index],
            jnp.asarray(rotations, dtype=precision_policy.score_real_dtype),
            image_shape,
            proj_volume_shape,
            disc_type,
            score_indices=window_indices,
            recon_indices=recon_window_indices,
            max_projected_rotations=_projection_cache_build_max_rotations_per_call(
                max_projected_rotations_per_projection_call,
                int(np.asarray(rotations).shape[0]),
            ),
            output_complex_dtype=precision_policy.score_complex_dtype,
            output_abs2_dtype=precision_policy.score_real_dtype,
            relion_projector_half=projection_halves[class_index],
            relion_projector_r_max=projection_r_max,
            projection_padding_factor=projection_padding_factor,
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
    # arrays gathered through a chunk-local slot; see _stream_chunk_projections.
    projection_bytes_per_rotation = transient_projection_bytes / float(max(n_projections, 1))
    # Both the whole-grid cache and the chunk-local caches are sized from what
    # the allocator can still hand out now, after reserving the half's resident
    # operands (allocated later): a fraction of the device total alone let the
    # whole-grid cache take memory that earlier iterations or a capped allocator
    # did not have (bench 14445196, K=1 10097 10k at hp3).
    physical_free_bytes = _device_free_memory_bytes()
    allocator_free_bytes = _jax_allocator_free_memory_bytes()
    pool_free_bytes = _jax_allocator_pool_free_bytes()
    # The operands are reserved only when the admission below would take them
    # at this reading: operands that stay per chunk are never allocated, and
    # reserving them anyway left a 0.38 GiB chunk budget at EMPIAR-10202
    # iteration 2 (box 800, bigbox 14446465).
    reserved_operand_bytes = 0
    # A tilt pass's CC iteration translates each image's resident CC operands per image slot (resident_tilts).
    if policy.resident_operands_requested() and (not firstiter_cc or tilt is not None):
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
    stream_projections = tilt is not None or not _projection_cache_fits_budget(
        cache_projection_bytes, stream_projection_budget_bytes
    )
    # Streamed chunks keep the per-chunk operand preparation. Their working set
    # (chunk-local projection caches and the posterior program's temporaries)
    # is not in the chunk estimate, and next to 21.5 GiB of resident operands
    # the full EMPIAR-10097 hp3 pass (65000 images, current size 126) could not
    # allocate 9.65 GiB (job 14518607); the per-chunk preparation ran that pass
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
        if _projection_cache_fits_budget(cache_projection_bytes, unreserved_budget_bytes):
            stream_projections = False
            operands_yield_to_cache = True
            stream_projection_budget_bytes = unreserved_budget_bytes
        reserved_operand_bytes = 0
    # The union cache is one array. The free totals above do not promise one block
    # that large (EMPIAR-10345 it13 half 2 could not allocate 20.19 GiB with over
    # 40 GiB free, bench 14643272), so it is allocated now, and a pass whose
    # allocator refuses it streams its projections instead. Deciding from the
    # allocator's largest free block instead streamed Class3D K4 100k from
    # iteration 19, a 5.50 GiB cache against a 5.25 GiB block, at 3.5x the
    # iteration time (bench 14719385).
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
        rows_per_call = _projection_cache_build_max_rotations_per_call(
            max_projected_rotations_per_projection_call, cache_rows_per_class
        ) or cache_rows_per_class

        def project_union_rows(class_index, start, stop):
            return _compute_sparse_pass2_windowed_projections_block(
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
                resident_stable_windows.pass_window_union(
                    sgd_pixels.astype(np.int64),
                    image_shape=image_shape,
                    projector_output_size=int(projection_kwargs["projector_output_size"]),
                    capacity_logical_output_size=capacity_logical_output_size,
                )
                if projection_kwargs.get("projector_output_size") is not None
                else None
            )
            sgd_projection_kwargs = {**projection_kwargs, "relion_kernel": "sgd"}

            def project_sgd_rows(class_index, start, stop):
                return _compute_sparse_pass2_windowed_projections_block(
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
    # when the cache holds one (_cached_block_projections), not the score
    # window's. Counting the score window let a 524288-row chunk past both
    # budgets at current size 108 and its 19.8 GiB gather ran out of memory
    # (union 5008 vs score 3690 pixels, job 14592696).
    gathered_row_pixels = int(n_windowed) if union_indices is None else int(union_indices.shape[0])
    # A row-blocked cache's gather holds the rows taken so far next to the next
    # block's (resident_scoring.cache_rows): two row copies instead of one.
    gathered_row_copies = 2 if isinstance(score_cache, tuple) else 1
    row_ladder = parse_env_capacity_ladder(_ROW_CAPACITY_LADDER_ENV, _DEFAULT_ROW_CAPACITY_LADDER)
    if stream_projections:
        row_ladder = resident_projections.stream_row_capacity_ladder(
            row_ladder,
            bytes_per_rotation=projection_bytes_per_rotation,
            max_projection_bytes=stream_projection_budget_bytes,
        )
    else:
        # The chunk gathers its rows out of the cache into one [rows, pixels]
        # block; at box 800 (10202, current size 304) capacity 131072 needs
        # 35.7 GiB for it (14434683). The budget is read again now that the
        # cache is resident; unknown readings do not cap.
        gather_budget_bytes = resident_projections.stream_projection_budget_bytes(
            device_memory_bytes if device_memory_bytes is not None else 1 << 62,
            physical_free_bytes=_device_free_memory_bytes(),
            allocator_free_bytes=_jax_allocator_free_memory_bytes(),
            pool_free_bytes=_jax_allocator_pool_free_bytes(),
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
    tilt_slot_block = None
    # Set by plan_chunks: whether a tilt pass runs its over-budget particles in row blocks.
    tilt_lone_row_blocks = False
    if tilt is not None:
        from relax.fine_pass.resident_tilts import (
            tilt_capacity_ladders,
            tilt_lone_row_bytes,
            tilt_projection_slot_block,
        )

        # A particle whose rows' projections over all its images do not fit is projected, scored and
        # backprojected a block of image slots at a time (resident_tilts.run_tilt_chunk).
        tilt_slot_block = tilt_projection_slot_block(
            resident_chunk_memory.max_image_rows(tables.row_offsets),
            slot_capacity=int(tilt.slot_capacity),
            slot_row_bytes=(
                resident_projections.STREAM_PEAK_COPIES * int(projection_bytes_per_rotation)
                if stream_projections
                else gathered_row_pixels * np.dtype(precision_policy.score_complex_dtype).itemsize
            )
            * gathered_row_copies,
            budget_bytes=resident_chunk_memory.resident_chunk_budget_bytes(reserved_bytes=reserved_operand_bytes),
        )
    whole_particle_slot_block = tilt_slot_block
    row_projection_factor = (1 if tilt is None else int(tilt_slot_block)) * gathered_row_copies
    if tilt is not None:

        row_ladder_start, _ = tilt_capacity_ladders(
            row_ladder_start, (1,), slot_capacity=int(tilt.slot_capacity)
        )
    # The accumulators already exist, so the budget reading counts them; the plan
    # records them so that its pass_bytes is the pass's total need.
    accumulator_bytes = resident_chunk_memory.resident_accumulator_bytes(
        int(np.prod(accumulator_shape)), recon_y_accum_dtype, recon_ctf_accum_dtype, n_slots=int(tables.n_slots)
    )

    def plan_chunks(unshifted_operands: bool):
        nonlocal tilt_slot_block, tilt_lone_row_blocks
        chunk_budget_bytes = resident_chunk_memory.resident_chunk_budget_bytes(reserved_bytes=reserved_operand_bytes)
        image_ladder = parse_env_capacity_ladder(_IMAGE_CAPACITY_LADDER_ENV, _DEFAULT_IMAGE_CAPACITY_LADDER)
        if tilt is not None or not (unshifted_operands and chunk_budget_bytes is not None):
            # The fixed translation-tile budget bounds the translated tiles; an
            # unshifted chunk holds only the Wavg rectangle and its exact
            # positions, which the joint plan below counts against the budget.
            # A tilt pass keeps its resident operands and this start (at S translations).
            image_ladder = resident_chunk_memory.resident_image_capacity_start(
                image_ladder,
                n_fine_trans=plan_translations,
                n_recon_pixels=n_recon_windowed,
                max_tile_bytes=_max_translation_tile_bytes_for_pass(
                    device_memory_bytes, has_external_normalization=False
                ),
                chunk_budget_bytes=chunk_budget_bytes,
            )
        mstep_block_rows = resident_chunk_memory.resolve_mstep_block_rows(
            n_recon_pixels=n_recon_windowed,
            max_block_bytes=_max_adjoint_block_bytes_for_pass(device_memory_bytes),
            row_capacity_ladder=row_ladder_start,
        )
        def plan_memory(lone_row_bytes, projection_factor):
            return resident_chunk_memory.plan_resident_chunk_memory(
                row_capacity_ladder=row_ladder_start,
                image_capacity_ladder=image_ladder,
                mstep_block_rows=mstep_block_rows,
                row_bytes=(
                    resident_projections.STREAM_PEAK_COPIES * int(projection_bytes_per_rotation)
                    if stream_projections
                    else gathered_row_pixels * np.dtype(precision_policy.score_complex_dtype).itemsize
                )
                * projection_factor,
                n_fine_trans=plan_translations,
                n_recon_pixels=n_recon_windowed,
                budget_bytes=chunk_budget_bytes,
                rows_live_during_prepare=True,
                pipelined=tilt is None and _global_chunk_loop_pipelined(stream_projections),
                float32_posterior_buckets=(
                    _FLOAT32_BUCKETED_IMAGE_SUMS
                    and tilt is None
                    and np.dtype(precision_policy.score_real_dtype) == np.dtype(np.float32)
                ),
                fixed_bytes=accumulator_bytes,
                max_image_rows=resident_chunk_memory.max_image_rows(tables.row_offsets),
                lone_row_bytes=lone_row_bytes,
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
            tilt_slot_block, tilt_lone_row_blocks = whole_particle_slot_block, False
            try:
                # A particle past the largest row class is one chunk holding all of its rows, when that fits.
                memory_plan = plan_memory(None, row_projection_factor)
            except ResidentConfigurationUnsupported:
                if firstiter_cc or not resident_chunk_memory.overflow_row_capacity(resident_chunk_memory.max_image_rows(tables.row_offsets), row_ladder_start):
                    raise
                # It does not: the particle runs alone in row blocks of the largest row class
                # (resident_tilts.run_tilt_chunk), which hold rows x slot block x pixels projection values
                # while every row keeps tilt_lone_row_bytes. The slot block is then sized for a block's rows,
                # not for the particle's.
                lone_slot_block = tilt_projection_slot_block(
                    max(int(v) for v in row_ladder_start),
                    slot_capacity=int(tilt.slot_capacity),
                    slot_row_bytes=(
                    resident_projections.STREAM_PEAK_COPIES * int(projection_bytes_per_rotation)
                    if stream_projections
                    else gathered_row_pixels * np.dtype(precision_policy.score_complex_dtype).itemsize
                )
                * gathered_row_copies,
                    budget_bytes=chunk_budget_bytes,
                )
                memory_plan = plan_memory(
                    tilt_lone_row_bytes(
                        int(n_fine_trans), slot_capacity=int(tilt.slot_capacity), n_accumulators=int(tables.n_slots)
                    ),
                    int(lone_slot_block) * gathered_row_copies,
                )
                tilt_slot_block, tilt_lone_row_blocks = int(lone_slot_block), True
                logger.info(
                    "Resident pass-2 tilt particles past the largest row class run in row blocks: the largest "
                    "(%d rows) does not fit one chunk; image slot block %d of %d",
                    resident_chunk_memory.max_image_rows(tables.row_offsets),
                    tilt_slot_block,
                    int(tilt.slot_capacity),
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
    # tiles held 32 images per chunk, 1560 chunks per half (job 14561455).
    chunk_unshifted_supported = _chunk_unshifted_operands_supported(
        bucket_io_kwargs,
        window_indices=window_indices,
        recon_window_indices=recon_window_indices,
        firstiter_cc=firstiter_cc,
        relion_native_fine_units=relion_native_fine_units,
        relion_exact_bpref_operands=relion_exact_bpref_operands,
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
    stats_config = resolve_statistics_config(
        n_shells=n_shells,
        n_fine_trans=n_fine_trans,
        n_images=n_units,
        n_coarse_rot=n_classes * n_coarse_rot,
        n_scale_groups=n_scale_groups,
        # The norm's unweighted cutoff is the weighted sums' size (no stable windows when it differs).
        current_size=program_current_size if wsum_current_size == current_size else wsum_current_size,
        include_unweighted_high_shell=include_unweighted_norm_high_shell,
        use_exact_relion_gaussian=use_exact_relion_gaussian,
        relion_wavg_atomic_direct_noise=relion_wavg_atomic_direct_noise,
        relion_wavg_atomic_scale_aa=relion_wavg_atomic_scale_aa,
        accumulate_scale=scale_groups_available,
        source_faithful_spectrum_norm=resolved_spectrum_norm,
        n_optics_groups=n_optics_groups,
        n_classes=n_classes,
        float32_bucketed_image_sums=_FLOAT32_BUCKETED_IMAGE_SUMS,
    )
    stats = make_resident_statistics(
        stats_config, max_posterior_dtype=precision_policy.score_real_dtype
    )
    image_tables = resident_image_terms.ChunkImageTables(
        shell_indices_half=jnp.asarray(shell_indices_half, dtype=jnp.int32),
        wavg_shell_indices=jnp.asarray(relion_wavg_rectangle.shell_indices, dtype=jnp.int32),
        wavg_scale_pixel_mask=jnp.asarray(scale_pixel_mask_rect_np, dtype=bool),
        translation_sqdist_ang=None,
    )
    window_logical = _window_logical_sizes(
        current_size=current_size,
        recon_pixels=logical_recon_pixels,
        rect_pixels=logical_rect_pixels,
        place=resident_chunk_records.PLACE_ON_DEVICE,
        wsum_current_size=None if wsum_current_size == current_size else wsum_current_size,
        # The capacity cube's adjoint clips at RELION's radius: its compact
        # trilinear bound and 3-D radius check read it (recovar backproject_indexed).
        mstep_max_r=None if stable_window_plan is None else _runtime_mstep_radius(mstep_max_r),
    )

    max_adjoint_block_bytes = _max_adjoint_block_bytes_for_pass(device_memory_bytes)
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
    if policy.resident_operands_requested() and (not firstiter_cc or tilt is not None):
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
        available_bytes = device_available_bytes(
            _device_free_memory_bytes(),
            _jax_allocator_free_memory_bytes(),
            _jax_allocator_pool_free_bytes(),
        )
        budget_bytes = resident_operands_max_bytes(available_bytes)
        if stream_keeps_chunk_operands:
            logger.info(
                "Resident pass-2 keeps the per-chunk operand preparation: %s, and one half's "
                "resident operands (%.2f GiB) would sit next to that working set",
                "the projections are streamed per chunk"
                if stream_projections
                else "the whole-grid projection cache fits only without them",
                operand_bytes / float(1024**3),
            )
        elif not resident_projections.resident_operands_fit(operand_peak_bytes, available_bytes):
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
            warm_config = resolve_compile_ahead_config()
            warm_pool = CompileAheadPool(warm_config)
            warm_predicted = None
            with warm_pool:
                # The warm-up describes the cached tables; a streamed pass keys
                # its programs on chunk-local tables, so it compiles in the loop.
                if warm_config.enabled and not stream_projections and n_classes == 1:
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
                        warmup = _submit_resident_chunk_warmup(
                            warm_pool,
                            # A lone overflow chunk runs its own row-blocked stages.
                            chunks=[chunk for chunk in chunks if not resident_chunk_memory.chunk_runs_alone(chunk, row_ladder)],
                            tables=tables,
                            n_fine_trans=int(n_fine_trans),
                            half_operand_avals=warm_predicted,
                            stage_tables=_make_chunk_stage_tables(
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
                    budget_bytes - operand_peak_bytes + resident_prepare_reserved_bytes(image_shape), image_shape
                )
                try:
                    resident_operands = prepare_resident_half_operands(
                        experiment_dataset, np.arange(n_images, dtype=np.int64), image_batch_size=prepare_batch,
                        **resident_operand_kwargs,
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
            if warm_config.enabled:
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
                    spec_difference = describe_chunk_spec_prediction(
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

    verify_operands = resident_operands is not None and policy.resident_operands_verify_enabled()

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
        from relax.fine_pass.resident_tilts import TiltMstepCensus, run_tilt_chunk

        # Without the half's resident operands (they do not fit), each chunk prepares the same
        # per-image operands for its own tilt images, a consecutive range of the half's images.
        if resident_operands is None:
            logger.info("Resident pass-2 tilt chunks prepare their own images' resident operands")
        tilt_base_tables = _make_chunk_stage_tables(
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
            n_classes=n_classes,
            # VDAM: each tilt image backprojects its residual (cuda_kernel_backproject3D_SGD per image).
            mstep_subtract_ctf_projection=bool(mstep_subtract_ctf_projection),
            firstiter_cc=bool(firstiter_cc),
        )
        unit_image_offsets = np.asarray(tilt.unit_image_offsets, dtype=np.int64)
        mstep_census = TiltMstepCensus()
        n_lone_chunks = 0
        for chunk in chunks:
            # An over-budget particle of a row-blocked plan (plan_chunks) runs in blocks of a row class.
            chunk_lone_block_rows = (
                resident_chunk_memory.lone_block_rows(chunk.row_capacity, row_ladder)
                if tilt_lone_row_blocks and resident_chunk_memory.chunk_runs_alone(chunk, row_ladder)
                else None
            )
            n_lone_chunks += int(chunk_lone_block_rows is not None)
            chunk_operands, operand_image_start = resident_operands, 0
            if chunk_operands is None:
                operand_image_start = int(unit_image_offsets[int(chunk.image_start)])
                operand_image_stop = int(unit_image_offsets[int(chunk.image_start) + int(chunk.n_valid_images)])
                chunk_operands = prepare_resident_half_operands(
                    experiment_dataset,
                    np.arange(operand_image_start, operand_image_stop, dtype=np.int64),
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
                tile_budget_bytes=_max_translation_tile_bytes_for_pass(
                    device_memory_bytes, has_external_normalization=False
                ),
                slot_block=tilt_slot_block,
                mstep_census=mstep_census,
                row_partial=row_partial,
                lone_block_rows=chunk_lone_block_rows,
                score_pixel_indices=None if window_indices is None else jnp.asarray(window_indices, dtype=jnp.int32),
            )
            Ft_y_total, Ft_ctf_total = Ft_y_chunk, Ft_ctf_chunk
        mstep_census.log()
        if n_lone_chunks:
            logger.info("Resident pass-2 ran %d lone overflow particle(s) in row blocks", n_lone_chunks)
    deferred = _global_chunk_loop_pipelined(stream_projections)
    pending = None
    pending_alone = False
    for chunk in chunks if tilt is None else ():
        alone = resident_chunk_memory.chunk_runs_alone(chunk, row_ladder)
        if pending is not None and (alone or pending_alone):
            Ft_y_total, Ft_ctf_total, stats = pending(Ft_y_total, Ft_ctf_total, stats)
            pending = None
        result = _run_resident_chunk(
            chunk,
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
    # RELION symmetriseReconstructions (ml_optimiser.cpp:5541-5575): x=0
    # Hermitian enforcement, then applyPointGroupSymmetry on BPref, per class.
    # C1 is the x=0 enforcement alone.
    # A stable-window C1 pass finalizes at its physical cube and crops the public
    # volume, so the finalize programs keep the physical class's shapes across
    # logical current sizes (they compiled at every logical size before).
    logical_recon_volume_shape = tuple(int(v) for v in recon_volume_shape)
    cropped = program_recon_volume_shape != logical_recon_volume_shape
    finalize_physical = cropped and physical_bpref_finalize_applies(
        program_recon_volume_shape, logical_recon_volume_shape, symmetry_label
    )
    finalize_shape = program_recon_volume_shape if finalize_physical else logical_recon_volume_shape
    Ft_y_out, Ft_ctf_out = [], []
    for class_Ft_y, class_Ft_ctf in zip(Ft_y_total, Ft_ctf_total):
        if cropped and not finalize_physical:
            class_Ft_y = crop_relion_x_half_accumulator(class_Ft_y, program_recon_volume_shape, recon_volume_shape)
            class_Ft_ctf = crop_relion_x_half_accumulator(
                class_Ft_ctf, program_recon_volume_shape, recon_volume_shape
            )
        class_Ft_y, class_Ft_ctf = finalize_half_volume_bpref(
            class_Ft_y,
            class_Ft_ctf,
            finalize_shape,
            logger=logger,
            label="Resident pass-2",
            symmetry_label=symmetry_label,
            relion_x_half=True,
        )
        class_Ft_y, class_Ft_ctf = relion_x_half_accumulators_to_public_layout(
            class_Ft_y,
            class_Ft_ctf,
            finalize_shape,
        )
        # ``keep_physical_bpref``: the caller slices the logical cube out of the physical one itself
        # (VDAM's accumulator adapter, on the host), so no crop program compiles per logical size.
        if finalize_physical and not keep_physical_bpref:
            class_Ft_y = crop_public_full_volume(class_Ft_y, program_recon_volume_shape, logical_recon_volume_shape)
            class_Ft_ctf = crop_public_full_volume(
                class_Ft_ctf, program_recon_volume_shape, logical_recon_volume_shape
            )
        Ft_y_out.append(class_Ft_y)
        Ft_ctf_out.append(class_Ft_ctf)

    finalized = finalize_statistics(stats, config=stats_config, n_images=n_units)
    noise_stats = make_noise_stats(
        wsum_sigma2_noise=finalized.wsum_sigma2_noise,
        wsum_img_power=finalized.wsum_img_power,
        wsum_sigma2_offset=finalized.wsum_sigma2_offset,
        sumw=finalized.sumw,
        wsum_norm_correction=finalized.wsum_norm_correction,
        wsum_scale_correction_xa=finalized.wsum_scale_correction_xa,
        wsum_scale_correction_aa=finalized.wsum_scale_correction_aa,
    )
    logger.info(
        "Resident pass-2: %d images, %d classes, %d chunks, %.2fs chunk loop, %.2fs total",
        n_images,
        n_classes,
        len(chunks),
        loop_s,
        time.time() - overall_t0,
    )
    return _ResidentPass2Result(
        Ft_y=tuple(Ft_y_out),
        Ft_ctf=tuple(Ft_ctf_out),
        n_classes=int(tables.n_classes),
        n_slot_groups=int(tables.n_slot_groups),
        finalized=finalized,
        noise_stats=noise_stats,
        fine_translations=fine_translations,
        score_real_dtype=precision_policy.score_real_dtype,
    )


def _with_reconstruction_groups(tables, group_ids, group_count, *, n_images: int):
    """Give each unit its M-step accumulator slot group (VDAM's pseudo-halfsets).

    ``group_ids`` is the caller's per-image group (VDAM: the subset schedule's
    pseudo-halfset, RELION's ``part_id % 2``, acc_ml_optimiser_impl.h:4800-4804);
    nothing is derived from dataset positions. ``group_count`` fixes the number of
    groups, so a group without images still has its (zero) accumulators.
    """

    if group_ids is None and group_count is None:
        return tables
    if group_ids is None or group_count is None:
        raise ValueError("reconstruction_group_ids and reconstruction_group_count go together")
    group_ids = np.asarray(group_ids, dtype=np.int32).reshape(-1)
    if group_ids.shape != (int(n_images),):
        raise ValueError(f"reconstruction_group_ids must have one entry per image ({n_images}), got {group_ids.shape}")
    return dataclass_replace(tables, unit_slot_offset=group_ids, n_slot_groups=int(group_count))


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


class _ResidentPass2Result(NamedTuple):
    """What :func:`_resident_pass2` hands its K=1 and K-class entries."""

    Ft_y: tuple  # per accumulator slot (class + K * group), public x-half layout
    Ft_ctf: tuple
    n_classes: int
    n_slot_groups: int
    finalized: FinalizedStatistics
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
    normalization_score_mode=None,
    return_score_log_z=False,
    return_score_log_z_only=False,
    rotation_block_size_for_quantization=5000,
    fine_source_eulers_override=None,
    return_source_eulers=False,
    fine_rotations_override=None,
    fine_mstep_rotations_override=None,
    fine_rotation_parent_override=None,
    fine_translations_override=None,
    fine_translation_parent_override=None,
    relion_half_volume_mstep=False,
    relion_x_half_mstep=False,
    mstep_subtract_ctf_projection=False,
    relion_fine_mstep_prune=False,
    relion_firstiter_score_mode="gaussian",
    relion_firstiter_winner_take_all=False,
    relion_exact_fine_gaussian=True,
    relion_fine_diff2_fused_ffi=False,
    relion_f32_fine_posterior=False,
    relion_exact_fine_normalized_cc=False,
    relion_projector_half=None,
    relion_projector_texture=None,
    relion_projector_r_max=None,
    adaptive_fraction=0.999,
    bpref_device_signature_active: bool = False,
    bpref_class_index: int = 0,
    include_unweighted_norm_high_shell: bool = True,
    preserve_bpref_particle_order: bool = False,
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
    adapter (``k_class._class_segmented_em_result``) turns either into the
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


_PLACE_AS_AVAL = resident_chunk_records.Placement(
    array=lambda value, dtype: jax.ShapeDtypeStruct(np.shape(value), jnp.dtype(dtype)),
    scalar=lambda value, dtype: jax.ShapeDtypeStruct((), jnp.dtype(dtype)),
    many=lambda items: {
        name: jax.ShapeDtypeStruct(np.shape(value), jnp.dtype(dtype)) for name, (value, dtype) in items.items()
    },
)


def _row_projection_ids(host_chunk, n_fine_rot: int | None, slot_of_projection=None) -> np.ndarray:
    """Each row's projection: its fine rotation, offset by ``class * n_fine_rot`` for K>1.

    The class-stacked projection caches and grids hold class ``k``'s fine grid at
    ``k * n_fine_rot``, so every consumer that gathers by a row's rotation id
    gathers its class's projection with the same statement. With a
    significant-parent cache (``slot_of_projection``, :class:`_ProjectionSlots`)
    the id is the projection's cache row instead; padded rows take row 0.
    """

    row_fine_rot = np.asarray(host_chunk["row_fine_rot"], dtype=np.int64)
    if n_fine_rot is not None:
        row_fine_rot = row_fine_rot + np.asarray(host_chunk["row_class"], dtype=np.int64) * int(n_fine_rot)
    if slot_of_projection is None:
        return row_fine_rot.astype(np.int32)
    n_valid_rows = int(host_chunk["n_valid_rows"])
    slots = np.zeros(row_fine_rot.shape, dtype=np.int32)
    slots[:n_valid_rows] = slot_of_projection[row_fine_rot[:n_valid_rows]]
    if bool(np.any(slots[:n_valid_rows] < 0)):
        raise RuntimeError("a candidate row's fine rotation is not in the significant-parent projection cache")
    return slots


def _chunk_class_layout(host_chunk, chunk, *, n_classes: int, n_fine_trans: int, place) -> resident_chunk_records.ChunkClassLayout:
    """The (slot, class) posterior sub-segments of one chunk."""

    n_segments = chunk.image_capacity * int(n_classes)
    row_class = np.asarray(host_chunk["row_class"], dtype=np.int64)
    row_segment = np.full(chunk.row_capacity, n_segments, dtype=np.int64)
    row_segment[:chunk.n_valid_rows] = (
        np.asarray(host_chunk["row_image_local"][:chunk.n_valid_rows], dtype=np.int64) * int(n_classes)
        + row_class[:chunk.n_valid_rows]
    )
    if np.any(np.diff(row_segment[:chunk.n_valid_rows]) < 0):
        raise ValueError("chunk rows must be image-major, then class-major")
    segment_rows = np.bincount(row_segment[:chunk.n_valid_rows], minlength=n_segments)
    segment_row_start = np.concatenate([[0], np.cumsum(segment_rows)])
    return resident_chunk_records.ChunkClassLayout(
        row_class=place.array(row_class, jnp.int32),
        row_segment=place.array(row_segment, jnp.int32),
        segment_offsets=place.array(segment_row_start * int(n_fine_trans), jnp.int32),
        segment_row_start=place.array(segment_row_start[:-1], jnp.int64),
    )


def _chunk_mstep_layout(host_chunk, *, place) -> resident_chunk_records.ChunkMstepLayout:
    """The chunk's accumulator slot per row (docs/development/resident_segments.md).

    Slot ``a = class + K * slot_offset[unit]``: a class's rows (Class3D, RELION's
    ``BPref[iclass]``, ml_optimiser.cpp:10826) and, for VDAM, a pseudo-halfset's
    (``iclass + (part_id % 2) * nr_classes``, acc_ml_optimiser_impl.h:4800-4804).
    The slot-major order itself is taken on the device, over the live rows only
    (:func:`_make_mstep_block_inputs`).
    """

    return resident_chunk_records.ChunkMstepLayout(row_slot=place.array(host_chunk["row_slot"], jnp.int32))


def _chunk_host_rows(tables: CandidateTableBlocks, chunk):
    """``(block tables, materialized host rows, chunk in block numbering)`` of one chunk.

    The rows are :func:`materialize_chunk`'s over the chunk's table block, with
    ``image_ids`` back in the pass's image numbering.
    """

    block_tables, local_chunk, image_base = tables.chunk_tables(chunk)
    host_chunk = materialize_chunk(block_tables, local_chunk)
    host_chunk["image_ids"][:chunk.n_valid_images] += np.int32(image_base)
    return block_tables, host_chunk, local_chunk


def _make_chunk_row_arrays(
    tables, chunk, n_fine_trans, *, place, n_fine_rot=None, slot_of_projection=None
) -> resident_chunk_records.ChunkRowArrays:
    """One chunk's row-aligned inputs, on the device or as avals.

    Everything here is host NumPy over the plan, so the aval placement costs
    only the materialize and does no device work at all. The shapes are the
    chunk's capacity class and nothing else, which
    ``tests/unit/test_chunk_row_avals.py`` pins: that is why warming one chunk
    per class covers every chunk of that class.

    A K-class table (``tables.n_classes > 1``) needs ``n_fine_rot``: its rows
    carry projection ids (:func:`_row_projection_ids`) and the class layout.
    """

    tables, host_chunk, local_chunk = _chunk_host_rows(tables, chunk)
    segment_offsets_np = chunk_segment_offsets(tables, local_chunk, n_fine_trans=n_fine_trans)
    image_row_start_np = segment_offsets_np.astype(np.int64)[:chunk.image_capacity] // int(n_fine_trans)
    image_row_count_np = (
        segment_offsets_np.astype(np.int64)[1:] - segment_offsets_np.astype(np.int64)[:-1]
    ) // int(n_fine_trans)
    if tables.n_classes > 1 and n_fine_rot is None:
        raise ValueError("a K-class chunk needs n_fine_rot for its projection ids")
    classes = None
    if tables.n_classes > 1:
        classes = _chunk_class_layout(
            host_chunk, chunk, n_classes=tables.n_classes, n_fine_trans=n_fine_trans, place=place
        )
    mstep = None
    if int(tables.n_slots) > 1:
        mstep = _chunk_mstep_layout(host_chunk, place=place)
    placed = place.many(
        {
            "row_image_local": (host_chunk["row_image_local"], jnp.int32),
            "row_fine_rot": (
                _row_projection_ids(host_chunk, n_fine_rot if tables.n_classes > 1 else None, slot_of_projection),
                jnp.int32,
            ),
            "row_log_prior": (host_chunk["row_log_prior"], jnp.float32),
            "row_mask_bits": (host_chunk["row_mask_bits"], jnp.uint32),
            "row_mask_mode": (host_chunk["row_mask_mode"], jnp.int8),
            "image_ids": (host_chunk["image_ids"], jnp.int32),
            "n_valid_rows": (host_chunk["n_valid_rows"], jnp.int32),
            "n_valid_images": (host_chunk["n_valid_images"], jnp.int32),
            "segment_offsets": (segment_offsets_np, jnp.int32),
            "image_row_start": (image_row_start_np, jnp.int64),
            "image_row_count": (image_row_count_np, jnp.int64),
        }
    )
    return resident_chunk_records.ChunkRowArrays(**placed, classes=classes, mstep=mstep)


def _make_chunk_translation_sqdist(
    default,
    *,
    translation_prior_centers_np,
    image_indices,
    image_capacity: int,
    n_valid_images,
    fine_translations,
    voxel_size,
):
    """The chunk's per-image prior squared distances, at image capacity.

    A per-image host table built at capacity keeps the program keyed on the
    capacity class and takes the eager device ops out of the chunk loop. Padded
    slots multiply a zero posterior, so their value is never observable; they
    are zeroed anyway.
    """

    if translation_prior_centers_np is None:
        return default
    padded_image_indices = _pad_batch_to_capacity(
        np.asarray(image_indices).reshape(-1, 1), image_capacity
    ).reshape(-1)
    centers = translation_prior_centers_for_images(
        translation_prior_centers_np,
        padded_image_indices,
        batch_size=image_capacity,
    )
    sqdist_np = np.asarray(translation_sqdist_angstrom(fine_translations, centers, voxel_size))
    sqdist_np = np.where(
        (np.arange(image_capacity) < int(n_valid_images))[:, None], sqdist_np, 0.0
    )
    return jnp.asarray(sqdist_np)


def _make_chunk_stage_operands(recon, translation_sqdist_ang) -> resident_chunk_records.ChunkStageOperands:
    """Name one chunk's operands out of whatever produced them.

    ``recon`` is the per-chunk preparation's dict, the once-per-half gather's
    dict, or -- for the compile-ahead warm-up -- the same gather's output under
    ``jax.eval_shape``, which is a dict of the same keys holding avals.
    """

    return resident_chunk_records.ChunkStageOperands(
        score_input=recon["score_input"],
        corr_img_score=recon["corr_img_score"],
        highres_xi2_half=recon["highres_xi2_half"],
        translation_prior=recon["translation_prior"],
        shifted_recon=recon.get("shifted_recon"),
        shifted_noise=recon.get("shifted_noise"),
        recon_image=recon.get("recon_image"),
        recon_weight=recon.get("recon_weight"),
        noise_image=recon.get("noise_image"),
        ctf2_over_nv_recon=recon["ctf2_over_nv_recon"],
        direct_ctf_rfloat_recon=recon["direct_ctf_rfloat_recon"],
        image_power_shells=recon["image_power_shells"],
        relion_norm_high_shell=recon["relion_norm_high_shell"],
        raw_translated_wavg_rectangle=recon["raw_translated_wavg_rectangle"],
        raw_translated_wavg_for_atomic=recon["raw_translated_wavg_for_atomic"],
        scale=recon["scale"],
        group_ids=recon["group_ids"],
        translation_sqdist_ang=translation_sqdist_ang,
        optics_groups=recon.get("optics_groups"),
        score_shifted_cc=recon.get("score_shifted_cc"),
        cc_half_batch_norm=recon.get("cc_half_batch_norm"),
        bpref_ctf2_over_nv_recon=recon.get("bpref_ctf2_over_nv_recon"),
    )


def chunk_program_path() -> str:
    """Which programs a chunk of this half will actually submit.

    ``"per-stage"`` is the default path's three glue programs, ``"eager"`` the
    loose dispatch with no program to warm. The compile-ahead warm-up reads this
    so it cannot warm a path the loop does not take. (A third path, one fused
    program per chunk, was removed in relax#76: nothing selected it.)
    """

    if policy.resident_glue_jit_enabled():
        return "per-stage"
    return "eager"


class _ChunkWarmup(NamedTuple):
    """What the warm-up queued, in the terms the chunk loop can be compared in."""

    classes: tuple
    keys: frozenset
    path: str


def describe_chunk_spec_prediction(
    *,
    predicted_rfloat_ctf_wavg,
    predicted_bpref_recon_operand,
    predicted_translate_sum_kernel,
    operands,
) -> str:
    """Name every spec boolean the warm-up predicted differently from the loop.

    Returns an empty string when they agree. The loop reads these three from the
    prepared operands; the warm-up has to predict them from configuration before
    the preparation runs, so this is the same predict-early verify-late check the
    operand tree gets, applied to the part of the program key that is not an aval.
    """

    if operands is None:
        return "the half prepared no resident operands, so no spec was used"
    problems = []
    for name, predicted, actual in (
        ("use_rfloat_ctf_wavg", bool(predicted_rfloat_ctf_wavg),
         operands.direct_ctf_rfloat_recon is not None),
        ("bpref_recon_operand", bool(predicted_bpref_recon_operand),
         operands.recon_weight is not None),
        ("use_translate_sum_kernel", bool(predicted_translate_sum_kernel), True),
    ):
        if predicted != actual:
            problems.append(f"{name}: predicted {predicted}, really {actual}")
    return "; ".join(problems)


def chunk_program_keys(path: str, spec) -> frozenset:
    """The ``(program name, spec)`` keys a chunk of ``spec`` will submit.

    A compiled program is identified by its function and its static argument,
    so this is the unit in which "what the warm-up compiled" and "what the loop
    ran" are the same kind of thing. Comparing capacity classes alone would
    miss a spec that differs in one of the booleans the warm-up has to predict
    from configuration, which is a real way for a warmed program to go unused.
    """

    return frozenset(
        (program.__name__, spec) for program in chunk_programs_for_path(path)
    )


def chunk_programs_for_path(path: str) -> tuple:
    """The jitted programs a chunk will submit on ``path``.

    One list, read by the compile-ahead warm-up and asserted against the
    runners by `tests/unit/test_em_compile_ahead_consumer.py`. Warming a
    program the runner does not submit, or missing one it does, costs compile
    time and buys nothing; that happened once, when a since-removed fused
    chunk program was warmed while the loop ran the per-stage one, and was found
    by reading a census rather than by a test.
    """

    if path == "per-stage":
        return (
            _resident_chunk_posterior_program,
            _resident_mstep_block_program,
            _resident_chunk_statistics_program,
        )
    if path == "eager":
        return ()
    raise ValueError(f"unknown chunk program path {path!r}")


@partial(jax.jit, static_argnames=("n_slots",))
def _make_mstep_block_inputs(rows, posterior, *, n_slots: int) -> "resident_chunk_records.MstepBlockInputs":
    """The M-step block program's row inputs, from the chunk and its posterior.

    Only rows the pruned posterior keeps enter the M-step. RELION backprojects
    and sums only weights at or above the significant weight
    (``significant_weight`` in collect2jobs and the backprojection kernel,
    acc_ml_optimiser_impl.h:4819); the fine posterior here already zeroes the
    rest, so a row with no positive cell adds exact zeros to every M-step
    accumulator. At the K4 100k/256 early state 5-10% of the scored rows are
    live, and the M-step was about 70% of the chunk time.

    The rows are taken slot-major (a single slot for K=1 refinement; a class
    for Class3D; a class and pseudo-halfset for VDAM, :class:`_ChunkMstepLayout`),
    live rows of a slot first in their chunk order, then every row that is not
    live; ``slot_offsets`` holds each slot's live run, so each slot is one run
    of blocks and the rows of a boundary block outside it get no weight. Grouping the live rows into fewer blocks changes
    which rows share a block, so the per-block partial sums are added in a
    different grouping; the contributions themselves are unchanged.

    Shared with the compile-ahead warm-up so the warmed signature is the one
    the per-stage loop submits. ``projections`` is None here: the block program
    gathers them from the tables itself.
    """

    row_is_live = posterior.row_is_valid & jnp.any(posterior.row_posterior > 0, axis=1)
    row_slot = jnp.zeros_like(rows.row_image_local) if rows.mstep is None else rows.mstep.row_slot
    key = jnp.where(row_is_live, row_slot, jnp.int32(n_slots)).astype(jnp.int32)
    order = jnp.argsort(key, stable=True)
    live_per_slot = jnp.bincount(key, length=int(n_slots) + 1)[: int(n_slots)]
    slot_offsets = jnp.concatenate(
        [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(live_per_slot).astype(jnp.int32)]
    )
    return resident_chunk_records.MstepBlockInputs(
        row_image_local=rows.row_image_local[order],
        kernel_row_image_ids=posterior.kernel_row_image_ids[order],
        row_posterior=posterior.row_posterior[order],
        row_fine_rot=rows.row_fine_rot[order],
        projections=None,
        slot_offsets=slot_offsets,
    )


def _make_chunk_program_spec(
    *,
    row_capacity,
    image_capacity,
    n_fine_trans,
    n_score_pixels,
    n_recon_pixels,
    n_rect,
    mstep_block_rows,
    adaptive_fraction,
    current_size,
    mstep_current_size,
    image_shape,
    recon_volume_shape,
    max_adjoint_block_bytes,
    stats_config,
    use_rfloat_ctf_wavg,
    use_translate_sum_kernel,
    bpref_recon_operand,
    mstep_max_r=None,
    reuse_coarse_normalization=False,
    firstiter_cc=False,
    n_slots=1,
    mstep_subtract_ctf_projection=False,
    n_classes=1,
    stable_window=False,
    union_native_fft_size=0,
    presum_adjoint=False,
) -> resident_chunk_records.ChunkProgramSpec:
    """The static key of one chunk program.

    The environment-read fields are the reason this is a function
    and not a literal at each call site: a warm-up that read them at a
    different moment, or not at all, would key its program differently from the
    loop's and warm nothing.
    """

    return resident_chunk_records.ChunkProgramSpec(
        row_capacity=int(row_capacity),
        image_capacity=int(image_capacity),
        n_fine_trans=int(n_fine_trans),
        n_score_pixels=int(n_score_pixels),
        n_recon_pixels=int(n_recon_pixels),
        n_rect=int(n_rect),
        mstep_block_rows=int(mstep_block_rows),
        adaptive_fraction=float(adaptive_fraction),
        current_size=int(current_size),
        mstep_current_size=int(mstep_current_size),
        image_shape=tuple(int(v) for v in image_shape),
        recon_volume_shape=tuple(int(v) for v in recon_volume_shape),
        max_adjoint_block_bytes=int(max_adjoint_block_bytes),
        stats_config=stats_config,
        use_rfloat_ctf_wavg=bool(use_rfloat_ctf_wavg),
        use_translate_sum_kernel=bool(use_translate_sum_kernel),
        bpref_recon_operand=bool(bpref_recon_operand),
        mstep_max_r=float(int(mstep_current_size) // 2) if mstep_max_r is None else mstep_max_r,
        kernel_ctf_probs=policy.kernel_ctf_probs_enabled(),
        reuse_coarse_normalization=bool(reuse_coarse_normalization),
        firstiter_cc=bool(firstiter_cc),
        n_slots=int(n_slots),
        mstep_subtract_ctf_projection=bool(mstep_subtract_ctf_projection),
        n_classes=int(n_classes),
        stable_window=bool(stable_window),
        union_native_fft_size=int(union_native_fft_size),
        presum_adjoint=bool(presum_adjoint),
    )


def _submit_resident_chunk_warmup(
    pool,
    *,
    chunks,
    tables,
    n_fine_trans,
    half_operand_avals,
    stage_tables,
    carry,
    translation_angles,
    rect_indices,
    exact_positions,
    image_shape,
    spec_kwargs,
    translation_prior_centers_np,
    fine_translations,
    voxel_size,
    default_translation_sqdist,
):
    """Queue the chunk programs of every capacity class the plan will run.

    Called in the window between the admission check and the per-half operand
    preparation. The preparation is 2.4-5.8 s of host-bound device dispatch on
    the main thread, and the chunk programs the loop will need after it are
    fully determined by then: the capacity classes are in ``chunks``, the
    iteration-global tables exist, and the operands the programs consume are
    described by ``half_operand_avals`` without being prepared.

    Nothing here can change a result. The warm-up hands the helper thread
    shape/dtype stand-ins only, and if it describes a program the loop does not
    run, the loop compiles its own as before; the cost is the wasted warm-up and
    the log line below is how that is noticed.

    Returns the capacity classes submitted, for the hit-rate line.
    """

    import dataclasses

    from relax.fine_pass.resident_operands import ResidentHalfOperands

    # Warm the programs the configured path will actually submit: the per-stage
    # path with the glue JIT on is what production runs, and it is three
    # programs. The path is decided here, before any work, and named in the log line.
    path = chunk_program_path()
    if path == "eager":
        return ()

    def as_aval(value):
        if value is None:
            return None
        return jax.ShapeDtypeStruct(
            tuple(int(d) for d in np.shape(value)), jnp.dtype(value.dtype)
        )

    expected_programs = chunk_programs_for_path(path)

    def _checked(work):
        # The warm-up must submit exactly the programs the runner for this path
        # submits. Raising here lands in the pool's own error record, so a
        # mismatch is reported and the run is untouched.
        got = tuple(program for program, _, _ in work)
        if set(got) != set(expected_programs):
            raise ValueError(
                "compile-ahead would warm "
                f"{sorted(p.__name__ for p in got)} on the {path} path, but that path "
                f"runs {sorted(p.__name__ for p in expected_programs)}"
            )
        return work

    table_avals = jax.tree_util.tree_map(as_aval, stage_tables)
    carry_avals = jax.tree_util.tree_map(as_aval, carry)
    angle_aval = as_aval(translation_angles)
    rect_aval = as_aval(rect_indices)
    exact_aval = as_aval(exact_positions)

    names = [
        f.name for f in dataclasses.fields(ResidentHalfOperands) if not f.name.startswith("n_")
    ]
    present = [n for n in names if getattr(half_operand_avals, n) is not None]
    scalars = {
        f.name: getattr(half_operand_avals, f.name)
        for f in dataclasses.fields(ResidentHalfOperands)
        if f.name.startswith("n_")
    }
    present_avals = [getattr(half_operand_avals, n) for n in present]

    submitted = []
    submitted_keys = set()
    for chunk in chunks:
        capacity_class = (int(chunk.row_capacity), int(chunk.image_capacity))
        if capacity_class in submitted:
            continue
        row_capacity, image_capacity = capacity_class
        row_avals = _make_chunk_row_arrays(tables, chunk, n_fine_trans, place=_PLACE_AS_AVAL)
        sqdist = _make_chunk_translation_sqdist(
            default_translation_sqdist,
            translation_prior_centers_np=translation_prior_centers_np,
            image_indices=np.arange(chunk.image_start, chunk.image_stop, dtype=np.int64),
            image_capacity=image_capacity,
            n_valid_images=chunk.n_valid_images,
            fine_translations=fine_translations,
            voxel_size=voxel_size,
        )
        spec = _make_chunk_program_spec(
            row_capacity=row_capacity, image_capacity=image_capacity, **spec_kwargs
        )

        def thunk(_row=row_avals, _spec=spec, _images=image_capacity, _sq=as_aval(sqdist)):
            # The chunk operands are predicted by tracing the real gather, not
            # by a second constructor: the gather's own shape validation runs on
            # the way through, and there is no place for the two to disagree.
            def gather(slots, angles, rect, exact, *arrays):
                fields = dict(scalars)
                fields.update({name: None for name in names})
                fields.update(dict(zip(present, arrays)))
                return gather_resident_chunk_operands(
                    ResidentHalfOperands(**fields),
                    slots,
                    translation_angles=angles,
                    rect_indices=rect,
                    exact_positions=exact,
                    image_shape=image_shape,
                )

            recon = jax.eval_shape(
                gather,
                jax.ShapeDtypeStruct((_images,), jnp.int32),
                angle_aval,
                rect_aval,
                exact_aval,
                *present_avals,
            )
            operand_avals = _make_chunk_stage_operands(recon, _sq)
            # The per-stage path runs three programs.
            # Their later inputs are earlier stages' outputs, so they are taken
            # by tracing those stages rather than described a second time.
            posterior_avals = jax.eval_shape(
                partial(_resident_chunk_posterior_program, spec=_spec),
                _row,
                operand_avals,
                table_avals,
            )
            mstep_avals = jax.eval_shape(
                partial(_initial_mstep_carry, spec=_spec),
                carry_avals[0][0],
                carry_avals[1][0],
                operand_avals,
                table_avals,
            )
            block_avals = jax.eval_shape(
                partial(_make_mstep_block_inputs, n_slots=int(_spec.n_slots)), _row, posterior_avals
            )
            return _checked([
                (
                    _resident_chunk_posterior_program,
                    (_row, operand_avals, table_avals),
                    {"spec": _spec},
                ),
                (
                    _resident_mstep_block_program,
                    (
                        jax.ShapeDtypeStruct((), jnp.int32),
                        block_avals,
                        operand_avals,
                        table_avals,
                        mstep_avals,
                    ),
                    {"spec": _spec},
                ),
                (
                    _resident_chunk_statistics_program,
                    (
                        carry_avals[2],
                        _row,
                        operand_avals,
                        table_avals,
                        posterior_avals,
                        mstep_avals,
                    ),
                    {"spec": _spec},
                ),
            ])

        if pool.submit_thunk(f"resident chunk rows={row_capacity} images={image_capacity}", thunk):
            submitted.append(capacity_class)
            submitted_keys.update(chunk_program_keys(path, spec))
    return _ChunkWarmup(classes=tuple(submitted), keys=frozenset(submitted_keys), path=path)


def _make_chunk_stage_tables(
    *,
    projection_score_cache,
    projection_recon_cache,
    projection_recon_abs2_cache,
    mstep_grid,
    coarse_parent_grid,
    fine_translation_parent_device,
    half_weights,
    translation_angles,
    full_to_compact,
    noise_variance_for_noise,
    shell_indices_noise,
    exact_positions_device,
    recon_pixel_indices,
    relion_x_half_recon_indices,
    image_tables,
    cache_slot_fine_rot=None,
    coarse_reuse=None,
    window_logical=None,
    union_score_take=None,
    union_recon_take=None,
    residual_sgd_cache=None,
    residual_sgd_take=None,
) -> resident_chunk_records.ChunkStageTables:
    """Assemble the iteration-global tables every chunk of a half reads.

    One builder, called by the chunk loop with the real arrays and by the
    compile-ahead warm-up with their shape/dtype stand-ins. Assembling the
    tuple twice would be a place for the warm-up to drift from the loop: a
    warmed program with one field's dtype wrong is never used, which costs
    compile time and is invisible unless someone reads the hit rate.
    """

    return resident_chunk_records.ChunkStageTables(
        projection_score_cache=projection_score_cache,
        projection_recon_cache=projection_recon_cache,
        projection_recon_abs2_cache=projection_recon_abs2_cache,
        mstep_grid=mstep_grid,
        coarse_parent_grid=coarse_parent_grid,
        fine_translation_parent=fine_translation_parent_device,
        half_weights=half_weights,
        translation_angles=translation_angles,
        full_to_compact=full_to_compact,
        noise_variance_for_noise=noise_variance_for_noise,
        shell_indices_noise=shell_indices_noise,
        exact_positions=exact_positions_device,
        recon_pixel_indices=recon_pixel_indices,
        relion_x_half_recon_indices=relion_x_half_recon_indices,
        shell_indices_half=image_tables.shell_indices_half,
        wavg_shell_indices=image_tables.wavg_shell_indices,
        wavg_scale_pixel_mask=image_tables.wavg_scale_pixel_mask,
        cache_slot_fine_rot=cache_slot_fine_rot,
        coarse_reuse=coarse_reuse,
        window_logical=window_logical,
        union_score_take=union_score_take,
        union_recon_take=union_recon_take,
        residual_sgd_cache=residual_sgd_cache,
        residual_sgd_take=residual_sgd_take,
    )


def _chunk_unshifted_operands_supported(
    bucket_io_kwargs,
    *,
    window_indices,
    recon_window_indices,
    firstiter_cc,
    relion_native_fine_units,
    relion_exact_bpref_operands,
) -> bool:
    """Whether a chunk can take its own images' unshifted operands (:func:`unshifted_chunk_operands`).

    The checks :func:`prepare_resident_half_operands` makes, before any image
    is prepared, so the chunks are sized for the operand family they will use.
    The ``--firstiter_cc`` iteration scores translated normalized-CC tiles,
    which only the translated preparation builds.
    """

    if not policy.resident_operands_requested() or firstiter_cc:
        return False
    try:
        require_unshifted_operand_support(
            bucket_io_kwargs, window_indices=window_indices, recon_window_indices=recon_window_indices
        )
        if relion_native_fine_units and not relion_exact_bpref_operands:
            raise ResidentOperandsUnsupported(
                "native-unit fine scores without RELION's RFLOAT CTF operand"
            )
    except ResidentOperandsUnsupported as reason:
        logger.info("Resident pass-2 chunks keep the translated tiles: %s", reason)
        return False
    return True


def unshifted_chunk_operands(
    experiment_dataset,
    image_indices,
    *,
    image_capacity: int,
    bucket_io_kwargs,
    window_indices,
    recon_window_indices,
    rect_indices_device,
    exact_positions_device,
    translation_angles,
    noise_shell_indices_half,
    n_noise_shells: int,
    image_shape,
    current_size,
    n_fine_trans: int,
    accumulate_noise,
    source_faithful_spectrum_norm: bool,
    fine_translation_prior_2d,
    scale_corrections_np,
    group_ids_np,
    optics_groups_np,
    precision_policy,
    use_exact_relion_gaussian: bool = True,
    relion_native_fine_units: bool = False,
):
    """One chunk's unshifted per-image operands, gathered at the chunk's image capacity.

    The once-per-half preparation (T16, :func:`prepare_resident_half_operands`)
    run for the chunk's own images, then its chunk gather: the M-step
    translates these operands inside T15's translate-and-sum kernel, which
    skips zero weights, instead of reducing a pre-shifted ``[images,
    translations, pixels]`` tile. Per chunk, not per half, so the operands
    never scale with the particle count. The padded slots carry -1 and the
    gather zeroes them, as the tile path's capacity mask did. Raises
    :class:`ResidentOperandsUnsupported` for a configuration the preparation
    does not cover.
    """

    operands = prepare_resident_half_operands(
        experiment_dataset,
        image_indices,
        bucket_io_kwargs=bucket_io_kwargs,
        window_indices=window_indices,
        recon_window_indices=recon_window_indices,
        wavg_rect_indices=rect_indices_device,
        noise_shell_indices_half=noise_shell_indices_half,
        n_noise_shells=int(n_noise_shells),
        image_shape=image_shape,
        current_size=current_size,
        n_fine_trans=int(n_fine_trans),
        use_exact_relion_gaussian=bool(use_exact_relion_gaussian),
        accumulate_noise=accumulate_noise,
        source_faithful_spectrum_norm=bool(source_faithful_spectrum_norm),
        fine_translation_prior_2d=fine_translation_prior_2d,
        scale_corrections_np=scale_corrections_np,
        group_ids_np=group_ids_np,
        precision_policy=precision_policy,
        image_batch_size=int(image_capacity),
        optics_groups_np=optics_groups_np,
        relion_native_fine_units=bool(relion_native_fine_units),
        log_summary=False,
    )
    n_images = int(np.asarray(image_indices).shape[0])
    image_slots = np.full(int(image_capacity), -1, dtype=np.int32)
    image_slots[:n_images] = np.arange(n_images, dtype=np.int32)
    return gather_resident_chunk_operands(
        operands,
        image_slots,
        translation_angles=translation_angles,
        rect_indices=rect_indices_device,
        exact_positions=exact_positions_device,
        image_shape=image_shape,
    )


def _prepare_chunk_reconstruction_operands(
    *,
    chunk,
    image_indices,
    experiment_dataset,
    bucket_io_kwargs,
    windowed_prepare,
    recon_window_indices,
    score_window_indices,
    fine_translation_prior_2d,
    score_real_dtype,
    n_fine_trans,
    n_recon_windowed,
    image_shape,
    current_size,
    use_exact_relion_gaussian,
    accumulate_noise,
    source_faithful_spectrum_norm,
    relion_score_translation_angles,
    rect_indices_device,
    exact_positions_device,
    scale_corrections_np,
    group_ids_np,
    optics_groups_np=None,
    relion_native_fine_units=False,
    normalized_cc=False,
    noise_shell_indices_half=None,
    n_noise_shells=None,
):
    """Build one chunk's translated reconstruction, noise and Wavg tiles.

    The oracle path, kept selectable by
    ``RELAX_SPARSE_PASS2_RESIDENT_OPERANDS=0``. These tiles carry the
    ``(images, translations, pixels)`` axis, so they are the one operand family
    that cannot be kept resident for a whole half (17 GiB at the hp3 state);
    they are rebuilt per chunk from the same :func:`_prepare_bucket_io` call,
    with the same keyword arguments, that the compact engine makes per bucket.
    The default path instead keeps the *unshifted* per-image operands resident
    (:mod:`recovar.em.sparse_pass2.resident_operands`) and lets T15's kernel
    apply the translations inside the M-step reduction, so no tile is built at
    all.

    ``relion_native_fine_units`` gives the score operands in RELION's native
    FFT units exactly as :func:`prepare_resident_half_operands` does: the score
    image divided by N**2 and RELION's native ``corr_img``. Resident local
    search calls this without it and keeps RECOVAR units.

    ``normalized_cc`` (RELION's ``--firstiter_cc`` iteration) also returns the
    compact engine's normalized-CC operands: the translated corrected score
    tile ``[C_B, T, P]`` and half the unweighted image power, the score offset
    the compact engine reports evidence with. ``corr_img_score`` is then the
    CC pixel weight the same call returns.

    Shape stability. The batch handed to ``_prepare_bucket_io`` is padded on
    the host to the chunk's image capacity before anything is traced, so every
    chunk of one capacity class runs the same program instead of one program
    per distinct occupancy. The padded slots carry a duplicate image's real
    data through preparation and are zeroed afterwards by a capacity-shaped
    mask. Nothing downstream reads them: their posterior rows are zero and
    their image ids are -1.
    """

    image_indices = np.asarray(image_indices)

    batch_images, padded_ctf_params, fetched_indices, padded_fetched_indices = fetch_capacity_batch(
        experiment_dataset, image_indices, chunk.image_capacity
    )
    order = _reorder_permutation(fetched_indices, image_indices, chunk.image_capacity)
    prepared = _prepare_bucket_io(
        experiment_dataset,
        batch_images,
        padded_ctf_params,
        padded_fetched_indices,
        return_direct_scoring_io=True,
        **bucket_io_kwargs,
    )
    (
        _shifted_score_half,
        shifted_recon_half,
        batch_norm,
        ctf2_over_nv_half,
        ctf2_over_nv_half_with_dc,
        shifted_score_half_with_dc,
        processed_score_half_for_noise,
        shifted_corrected_score_half,
        direct_score_input,
        _direct_preprocessed_score_input,
        _direct_pixel_correction,
        _direct_preprocess_normalization_factors,
        _direct_integer_pre_shifts,
        _direct_batch_image_corrections,
        direct_batch_scale_corrections,
        _direct_inverse_noise_half,
        direct_ctf_rfloat_half,
    ) = prepared

    gather_recon = jnp.asarray(recon_window_indices, dtype=jnp.int32)
    if windowed_prepare:
        shifted_recon = shifted_recon_half
        ctf2_over_nv_recon = ctf2_over_nv_half_with_dc
        shifted_noise = shifted_score_half_with_dc
    else:
        shifted_recon = shifted_recon_half[:, gather_recon]
        ctf2_over_nv_recon = ctf2_over_nv_half_with_dc[:, gather_recon]
        shifted_noise = shifted_score_half_with_dc[:, gather_recon]
    direct_ctf_rfloat_recon = (
        None if direct_ctf_rfloat_half is None else direct_ctf_rfloat_half[:, gather_recon]
    )
    bpref_ctf2_over_nv_recon = None
    ctf_premultiplied = ctf.premultiplied_ctf_rows(experiment_dataset, padded_fetched_indices, image_shape)
    if ctf_premultiplied is not None:
        # The same weights prepare_unshifted_bucket_operands gives the resident operands.
        _, premultiplied_ctf_weight = premultiplied_bpref_weights(
            jnp.asarray(direct_ctf_rfloat_half, dtype=ctf2_over_nv_recon.dtype),
            _direct_inverse_noise_half,
            (
                jnp.asarray(direct_batch_scale_corrections, dtype=ctf2_over_nv_recon.dtype)
                if bucket_io_kwargs["scale_corrections"] is not None
                else None
            ),
        )
        bpref_ctf2_over_nv_recon = jnp.where(
            jnp.asarray(ctf_premultiplied)[:, None], premultiplied_ctf_weight[:, gather_recon], ctf2_over_nv_recon
        )

    # Score-side operands come from this same call. The driver used to make a
    # second pass over the whole half for them; the preparation is per-image
    # pure (bitwise at batch 256, 128, 32, 13 and 1), so taking them here is
    # the same arithmetic with one call per chunk instead of two per image.
    if windowed_prepare:
        score_input = direct_score_input
        corr_img_score = ctf2_over_nv_half
    else:
        gather_score = jnp.asarray(score_window_indices, dtype=jnp.int32)
        score_input = direct_score_input[:, gather_score]
        corr_img_score = ctf2_over_nv_half[:, gather_score]
    if relion_native_fine_units:
        if direct_ctf_rfloat_half is None:
            raise ValueError("native-unit fine scores require RELION's RFLOAT CTF operand")
        # Same operands as the once-per-half preparation and the compact
        # engine: RELION's native corr_img (DC zeroed for half-spectrum
        # scoring) in the score window, and the unshifted score image divided
        # by N**2, which the kernel translates in-kernel.
        native_corr_img_half = _relion_native_score_corr_img(
            pixel_rows(
                noise_rows(
                    bucket_io_kwargs["noise_variance_half"],
                    bucket_io_kwargs.get("noise_optics_groups"),
                    padded_fetched_indices,
                )
            ),
            direct_ctf_rfloat_half,
            image_shape,
            (
                jnp.asarray(direct_batch_scale_corrections, dtype=jnp.float32)[:, None]
                if bucket_io_kwargs["scale_corrections"] is not None
                else None
            ),
            zero_dc=bool(bucket_io_kwargs["half_spectrum_scoring"]),
        )
        if score_window_indices is not None:
            native_corr_img_half = native_corr_img_half[
                :, jnp.asarray(score_window_indices, dtype=jnp.int32)
            ]
        corr_img_score = native_corr_img_half.astype(corr_img_score.dtype)
        score_input = _relion_native_fine_units(score_input, int(np.prod(image_shape)))

    highres_xi2_half, relion_norm_high_shell = _relion_powerclass_noise_terms(
        processed_score_half_for_noise,
        image_shape=image_shape,
        current_size=current_size,
        use_exact_relion_gaussian=use_exact_relion_gaussian,
        accumulate_noise=accumulate_noise,
        source_faithful_spectrum_norm=source_faithful_spectrum_norm,
    )
    raw_translated_wavg_rectangle = _relion_cuda_translate_wavg_norm_images(
        processed_score_half_for_noise,
        relion_score_translation_angles,
        rect_indices_device,
        image_shape,
    )

    permutation = jnp.asarray(order, dtype=jnp.int32)
    valid_images = jnp.asarray(
        np.arange(chunk.image_capacity) < chunk.n_valid_images, dtype=bool
    )
    score_shifted_cc = cc_half_batch_norm = None
    if normalized_cc:
        # The compact engine's operands (the deleted sparse_pass2_bucketed.py, the
        # normalized_cc branch): the translated corrected score tile in the
        # score window, and the -0.5 * |image|^2 evidence offset.
        tile = shifted_corrected_score_half.reshape(chunk.image_capacity, int(n_fine_trans), -1)
        if not windowed_prepare:
            tile = tile[:, :, jnp.asarray(score_window_indices, dtype=jnp.int32)]
        score_shifted_cc = _zero_padded_images(tile[permutation], valid_images)
        cc_half_batch_norm = _zero_padded_images(
            (0.5 * jnp.reshape(batch_norm, (chunk.image_capacity,)).real)[permutation], valid_images
        )

    (
        score_input,
        corr_img_score,
        highres_xi2_half,
        shifted_recon,
        shifted_noise,
        ctf2_over_nv_recon,
        direct_ctf_rfloat_recon,
        processed_image_half,
        relion_norm_high_shell,
        raw_translated_wavg_rectangle,
        raw_translated_wavg_for_atomic,
        bpref_ctf2_over_nv_recon,
    ) = _chunk_operand_rows(
        _ChunkOperandRowInputs(
            score_input=score_input,
            corr_img_score=corr_img_score,
            highres_xi2_half=highres_xi2_half,
            shifted_recon=shifted_recon,
            shifted_noise=shifted_noise,
            ctf2_over_nv_recon=ctf2_over_nv_recon,
            direct_ctf_rfloat_recon=direct_ctf_rfloat_recon,
            processed_score_half_for_noise=processed_score_half_for_noise,
            relion_norm_high_shell=relion_norm_high_shell,
            raw_translated_wavg_rectangle=raw_translated_wavg_rectangle,
            bpref_ctf2_over_nv_recon=bpref_ctf2_over_nv_recon,
        ),
        permutation,
        valid_images,
        exact_positions_device,
        image_capacity=int(chunk.image_capacity),
        n_fine_trans=int(n_fine_trans),
    )
    if int(shifted_recon.shape[-1]) != int(n_recon_windowed):
        raise ValueError(
            "reconstruction tile pixel count does not match the reconstruction window: "
            f"{int(shifted_recon.shape[-1])} vs {int(n_recon_windowed)}"
        )

    # Padded slots keep scale 1 so the Wavg kernel never divides by zero; their
    # posterior is zero, so the value is never observable.
    scale_chunk = np.ones(chunk.image_capacity, dtype=np.float32)
    if scale_corrections_np is not None:
        scale_chunk[:chunk.n_valid_images] = np.asarray(
            scale_corrections_np[image_indices], dtype=np.float32
        )
    group_ids_chunk = np.full(chunk.image_capacity, -1, dtype=np.int32)
    if group_ids_np is not None:
        group_ids_chunk[:chunk.n_valid_images] = np.asarray(
            group_ids_np[image_indices], dtype=np.int32
        )
    optics_groups_chunk = None
    if optics_groups_np is not None:
        optics_groups_chunk = np.zeros(chunk.image_capacity, dtype=np.int32)
        optics_groups_chunk[:chunk.n_valid_images] = np.asarray(
            optics_groups_np[image_indices], dtype=np.int32
        )

    translation_prior = jnp.asarray(
        np.zeros((chunk.image_capacity, int(n_fine_trans)), dtype=np.float32)
        if fine_translation_prior_2d is None
        else _pad_batch_to_capacity(
            np.asarray(fine_translation_prior_2d)[image_indices], chunk.image_capacity
        ),
        dtype=score_real_dtype,
    )

    if noise_shell_indices_half is None or n_noise_shells is None:
        raise ValueError("the chunk operands need the noise-shell binning of the packed half")
    return {
        "score_input": score_input,
        "corr_img_score": corr_img_score,
        "highres_xi2_half": highres_xi2_half,
        "translation_prior": _zero_padded_images(translation_prior, valid_images),
        "shifted_recon": shifted_recon,
        "shifted_noise": shifted_noise,
        "ctf2_over_nv_recon": ctf2_over_nv_recon,
        "direct_ctf_rfloat_recon": direct_ctf_rfloat_recon,
        "image_power_shells": image_power_shells(
            processed_image_half,
            jnp.asarray(noise_shell_indices_half, dtype=jnp.int32),
            shell_count=int(n_noise_shells),
        ),
        "relion_norm_high_shell": relion_norm_high_shell,
        "raw_translated_wavg_rectangle": raw_translated_wavg_rectangle,
        "raw_translated_wavg_for_atomic": raw_translated_wavg_for_atomic,
        "scale": jnp.asarray(scale_chunk),
        "group_ids": jnp.asarray(group_ids_chunk),
        "optics_groups": None if optics_groups_chunk is None else jnp.asarray(optics_groups_chunk),
        "score_shifted_cc": score_shifted_cc,
        "cc_half_batch_norm": cc_half_batch_norm,
        "bpref_ctf2_over_nv_recon": bpref_ctf2_over_nv_recon,
    }


def _global_chunk_loop_pipelined(stream_projections: bool) -> bool:
    """Whether the global chunk loop overlaps chunk k+1's front with chunk k's M-step.

    The chunk plan counts the second chunk's arrays (:func:`resident_chunk_bytes`)
    exactly when the loop runs this way.
    """

    return not policy.chunk_timing_enabled() and not stream_projections


def _verify_resident_chunk_operands(
    resident_recon,
    reference_recon,
    *,
    translation_angles,
    recon_pixel_indices,
    image_shape,
    n_recon_pixels: int,
    bpref_recon_operand: bool,
    label: str,
) -> None:
    """Prove one real chunk's resident operands equal the per-chunk preparation.

    The two paths differ in *when* the translation is applied, so the check
    translates the resident per-image operands with the primitives
    ``_prepare_bucket_io`` uses and compares against its own pre-shifted tiles.
    A mismatch raises: this runs only in a diagnostic arm, and a silent
    difference here would be a changed reconstruction operand.
    """

    from relax.cuda import translate as cuda_translate

    image_capacity = int(np.asarray(reference_recon["shifted_recon"]).shape[0])
    angles = jnp.asarray(translation_angles, dtype=jnp.float32)
    indices = jnp.asarray(recon_pixel_indices, dtype=jnp.int32)
    if bpref_recon_operand:
        translated_recon = cuda_translate.relion_translate_bpref_f32(
            jnp.asarray(resident_recon["recon_image"], dtype=jnp.complex64),
            jnp.asarray(resident_recon["recon_weight"], dtype=jnp.float32),
            angles,
            indices,
            image_shape,
        )
    else:
        translated_recon = cuda_translate.relion_translate_score_f32(
            jnp.asarray(resident_recon["recon_image"], dtype=jnp.complex64),
            angles,
            indices,
            image_shape,
        )
    translated_noise = cuda_translate.relion_translate_score_f32(
        jnp.asarray(resident_recon["noise_image"], dtype=jnp.complex64),
        angles,
        indices,
        image_shape,
    )
    n_fine_trans = int(angles.shape[0])
    checks = {
        "shifted_recon": (
            np.asarray(translated_recon).reshape(image_capacity, n_fine_trans, n_recon_pixels),
            np.asarray(reference_recon["shifted_recon"]),
        ),
        "shifted_noise": (
            np.asarray(translated_noise).reshape(image_capacity, n_fine_trans, n_recon_pixels),
            np.asarray(reference_recon["shifted_noise"]),
        ),
    }
    for name in (
        "score_input",
        "corr_img_score",
        "highres_xi2_half",
        "translation_prior",
        "ctf2_over_nv_recon",
        "direct_ctf_rfloat_recon",
        "image_power_shells",
        "relion_norm_high_shell",
        "raw_translated_wavg_rectangle",
        "raw_translated_wavg_for_atomic",
        "scale",
        "group_ids",
        "optics_groups",
    ):
        expected = reference_recon.get(name)
        actual = resident_recon.get(name)
        if expected is None and actual is None:
            continue
        if (expected is None) != (actual is None):
            raise AssertionError(f"resident operand {name} presence differs from the per-chunk path")
        checks[name] = (np.asarray(actual), np.asarray(expected))

    # ``relion_norm_high_shell`` bins the image power with a scatter-add over
    # duplicate indices. That races: two calls on the same array in one process
    # differ by about 6e-8 relative, so no two preparations of it are bitwise,
    # including two of the per-chunk path. It is checked against that spread
    # here, and exactly under ``RELAX_EM_DETERMINISTIC_REDUCTIONS=1``, where
    # the binning becomes a fixed-order masked reduction.
    racing_scatter = () if deterministic_reductions_enabled() else ("relion_norm_high_shell",)
    mismatched = []
    for name, (actual, expected) in checks.items():
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            mismatched.append(f"{name}: {actual.shape}/{actual.dtype} vs {expected.shape}/{expected.dtype}")
            continue
        if np.array_equal(actual, expected):
            continue
        differing = int(np.count_nonzero(actual != expected))
        left = actual.astype(np.complex128)
        right = expected.astype(np.complex128)
        worst = float(np.max(np.abs(left - right)))
        relative = float(
            np.max(np.abs(left - right) / np.maximum(np.abs(right), 1e-30))
        )
        if name in racing_scatter and relative <= _RACING_SCATTER_RELATIVE_BAND:
            logger.info(
                "Resident pass-2 operand verification on %s: %s is inside its racing "
                "scatter-add band (%d/%d cells, max |delta| %.3e, max relative %.3e); "
                "set RELAX_EM_DETERMINISTIC_REDUCTIONS=1 for an exact check",
                label, name, differing, actual.size, worst, relative,
            )
            continue
        mismatched.append(
            f"{name}: {differing}/{actual.size} cells differ, max |delta| {worst:.3e}, "
            f"max relative {relative:.3e}"
        )
    if mismatched:
        raise AssertionError(
            f"resident per-half operands differ from the per-chunk preparation on {label}: "
            + "; ".join(mismatched)
        )
    logger.info(
        "Resident pass-2 operand verification on %s: %d operands equal to the per-chunk "
        "preparation (BPref reconstruction operand=%d, deterministic reductions=%d)",
        label,
        len(checks),
        int(bpref_recon_operand),
        int(deterministic_reductions_enabled()),
    )


@jax.jit
def _times_logical_mask(values, n_logical):
    """``values`` times the mask of its first ``n_logical`` pixels (last axis); one program per physical class.

    The count is traced, so the logical sizes of a physical class share the program.
    """

    mask = jnp.arange(values.shape[-1], dtype=jnp.int32) < n_logical
    return values * mask.astype(values.dtype)


@partial(jax.jit, static_argnames=("shell_sentinel",))
def _mask_recon_tail(shell_indices_noise, noise_variance_for_noise, n_logical, *, shell_sentinel: int):
    """The recon tail past the ``n_logical`` pixels: the shell sentinel and zero noise variance.

    ``noise_variance_for_noise`` is ``[P]`` or, with optics groups, ``[G, P]``; the tail
    mask broadcasts over groups. One program per physical class (the count is traced).
    """

    tail = ~(jnp.arange(shell_indices_noise.shape[-1], dtype=jnp.int32) < n_logical)
    return (
        jnp.where(tail, jnp.int32(shell_sentinel), shell_indices_noise),
        jnp.where(tail, jnp.zeros((), noise_variance_for_noise.dtype), noise_variance_for_noise),
    )


# ---------------------------------------------------------------------------
# The chunk's stages and their program spec
# ---------------------------------------------------------------------------


class _MstepOnlyStatsConfig(NamedTuple):
    """The statistics fields the M-step block body reads.

    ``run_resident_mstep_blocks`` runs the M-step alone for a caller that owns
    its own scoring, posterior and statistics (local search, T12). Handing it
    these fields in the shape ``_ChunkProgramSpec`` expects keeps one M-step
    body without making that caller build a full statistics configuration.
    The local search sums its norm rows into their images with the scatter-add.
    """

    n_shells: int
    n_optics_groups: int = 1
    float32_bucketed_image_sums: bool = False


def _window_logical_sizes(
    *, current_size, recon_pixels, rect_pixels, place, mstep_max_r=None, wsum_current_size=None
) -> resident_chunk_records.WindowLogicalSizes:
    return resident_chunk_records.WindowLogicalSizes(
        current_size=place.scalar(int(current_size), jnp.int32),
        recon_pixels=place.scalar(int(recon_pixels), jnp.int32),
        rect_pixels=place.scalar(int(rect_pixels), jnp.int32),
        mstep_max_r=None if mstep_max_r is None else place.scalar(float(mstep_max_r), jnp.float32),
        wsum_current_size=None if wsum_current_size is None else place.scalar(int(wsum_current_size), jnp.int32),
    )


@jax.jit
def _fold_class_scale_sums_program(triplet, scale_xa_per_image, scale_aa_per_image, class_masks_rect, class_index):
    """The fold's arithmetic on the three arrays it changes: ``(triplet, scale_xa, scale_aa)``."""

    mask = jnp.asarray(class_masks_rect[class_index], dtype=bool).reshape(1, -1)
    zero = jnp.float32(0.0)
    xa = jnp.sum(jnp.where(mask, triplet[:, :, 0], zero).astype(jnp.float64), axis=1)
    aa = jnp.sum(jnp.where(mask, triplet[:, :, 1], zero).astype(jnp.float64), axis=1)
    return triplet.at[:, :, :2].set(zero), scale_xa_per_image + xa, scale_aa_per_image + aa


def _fold_class_scale_sums(mstep: "resident_chunk_records.ChunkMstepCarry", class_masks_rect, class_index) -> "resident_chunk_records.ChunkMstepCarry":
    """Move one class's Wavg XA/AA into the per-image scale sums under its own mask.

    ``class_masks_rect`` is every class's mask and ``class_index`` a device (or
    traced) index: selecting the row inside the program keeps the per-slot
    selection out of eager dispatch, one program for every class.

    RELION keeps XA and AA per class and adds them to the particle's scale sums
    only where that class's ``data_vs_prior_class > 3``
    (acc_ml_optimiser_impl.h:4893-4912); the diff2 channel is summed over
    classes. The class's blocks have just accumulated its XA/AA pixels, so they
    are masked and summed here and the two channels cleared for the next class.

    Only the triplet and the two scale vectors pass through the program; the rest of the carry is
    rebuilt around them. A program that took and returned the whole carry copied the slot's
    BPref accumulators (Ft_y and Ft_ctf) at every call: 2.37 GiB at a multi-optics K=2 Class3D
    slot of current size 114, which ran a 16 GB card out of memory (relax#36).
    """

    triplet, scale_xa, scale_aa = _fold_class_scale_sums_program(
        mstep.wavg_triplet_pixels, mstep.scale_xa_per_image, mstep.scale_aa_per_image, class_masks_rect, class_index
    )
    return mstep._replace(wavg_triplet_pixels=triplet, scale_xa_per_image=scale_xa, scale_aa_per_image=scale_aa)


def _logical_current_size(tables: resident_chunk_records.ChunkStageTables, spec: resident_chunk_records.ChunkProgramSpec):
    if tables.window_logical is None:
        return jnp.asarray(spec.current_size, dtype=jnp.int32)
    return tables.window_logical.current_size


def _resident_chunk_posterior(
    rows: resident_chunk_records.ChunkRowArrays,
    operands: resident_chunk_records.ChunkStageOperands,
    tables: resident_chunk_records.ChunkStageTables,
    *,
    spec: resident_chunk_records.ChunkProgramSpec,
) -> resident_chunk_records.ChunkPosterior:
    """Gather, cached projection, score and the segmented RELION posterior.

    Identical calls to the per-stage path; factored out so the jitted program
    and its oracle cannot drift apart.
    """

    row_index = jnp.arange(spec.row_capacity, dtype=jnp.int32)
    row_is_valid = row_index < rows.n_valid_rows
    kernel_row_image_ids = jnp.where(row_is_valid, rows.row_image_local, jnp.int32(-1))
    image_index = jnp.arange(spec.image_capacity, dtype=jnp.int32)
    chunk_image_ids = jnp.where(image_index < rows.n_valid_images, image_index, jnp.int32(-1))

    if spec.firstiter_cc:
        return _resident_chunk_posterior_firstiter_cc(
            rows,
            operands,
            tables,
            spec=spec,
            row_is_valid=row_is_valid,
            kernel_row_image_ids=kernel_row_image_ids,
        )
    scored = score_resident_chunk(
        rows.row_image_local,
        rows.row_fine_rot,
        rows.row_log_prior,
        rows.row_mask_bits,
        rows.row_mask_mode,
        rows.n_valid_rows,
        chunk_image_ids,
        tables.projection_score_cache,
        operands.score_input,
        operands.corr_img_score,
        operands.highres_xi2_half,
        operands.translation_prior,
        half_weights=tables.half_weights,
        translation_angles=tables.translation_angles,
        full_to_compact=tables.full_to_compact,
        fine_translation_parent=tables.fine_translation_parent,
        logical_current_size=_logical_current_size(tables, spec),
        row_capacity=spec.row_capacity,
        image_capacity=spec.image_capacity,
        n_fine_trans=spec.n_fine_trans,
        n_score_pixels=int(spec.n_score_pixels),
        score_take=tables.union_score_take,
        native_fft_size=int(spec.union_native_fft_size),
    )
    return _chunk_posterior_from_scores(
        scored,
        rows,
        tables,
        spec=spec,
        row_is_valid=row_is_valid,
        kernel_row_image_ids=kernel_row_image_ids,
    )


def _chunk_posterior_from_scores(
    scored,
    rows: resident_chunk_records.ChunkRowArrays,
    tables: resident_chunk_records.ChunkStageTables,
    *,
    spec: resident_chunk_records.ChunkProgramSpec,
    row_is_valid,
    kernel_row_image_ids,
) -> resident_chunk_records.ChunkPosterior:
    """The segmented RELION posterior of a scored chunk (:func:`_resident_chunk_posterior`'s tail)."""

    from relax.cuda import posterior as cuda_posterior

    scores_flat = jnp.asarray(scored.scores, dtype=jnp.float32).reshape(-1)

    reuse = bool(spec.reuse_coarse_normalization)
    if reuse:
        # rows.image_ids are the chunk's global image ids, -1 in padded slots,
        # whose values the segmented kernels never read.
        image_slot = jnp.maximum(rows.image_ids, jnp.int32(0))
        coarse_sum_weight = tables.coarse_reuse.sum_weight[image_slot]
        external_sum_weight = coarse_sum_weight.astype(jnp.float32)
    else:
        external_sum_weight = jnp.ones((spec.image_capacity,), dtype=jnp.float32)
    log_z = cuda_posterior.sparse_pass2_segmented_log_z_f64(
        scores_flat, rows.segment_offsets, rows.n_valid_images
    )
    posterior = cuda_posterior.sparse_pass2_segmented_posterior_f32(
        scores_flat,
        rows.segment_offsets,
        rows.n_valid_images,
        log_z,
        external_sum_weight,
        adaptive_fraction=float(spec.adaptive_fraction),
        keep_all=reuse,
        use_external_sum_weight=reuse,
    )
    (
        log_z_out,
        best_log_score,
        best_cell_index,
        max_posterior,
        _probs,
        _normalized_weights,
        reconstruction_probs,
        _mask,
        _n_significant,
        _sum_weight,
        _threshold,
    ) = posterior
    if reuse:
        # The compact engine's zero-oversampling arithmetic
        # (the deleted sparse_pass2_bucketed.py, reuse_coarse_normalization): RELION keeps the
        # coarse numeric sum but the fine pass's own exponent shift,
        # dLL = log(sum_weight) - (50 - best), and the coarse winner and Pmax.
        exponent_add = jnp.float32(50.0) - jnp.asarray(best_log_score, dtype=jnp.float32)
        log_z_out = jnp.log(coarse_sum_weight.astype(jnp.float64)) - exponent_add.astype(jnp.float64)
        best_cell_index = tables.coarse_reuse.winner_cell[image_slot]
        max_posterior = tables.coarse_reuse.max_posterior[image_slot].astype(
            jnp.asarray(max_posterior).dtype
        )
    classes = None
    if rows.classes is not None:
        classes = _class_sub_segment_posterior(
            scores_flat.reshape(spec.row_capacity, spec.n_fine_trans),
            rows,
            row_is_valid,
            n_segments=spec.image_capacity * int(spec.n_classes),
            n_classes=int(spec.n_classes),
        )
    return resident_chunk_records.ChunkPosterior(
        row_posterior=jnp.asarray(reconstruction_probs, dtype=jnp.float32).reshape(
            spec.row_capacity, spec.n_fine_trans
        ),
        min_diff2=scored.min_diff2,
        class_log_z=jnp.asarray(log_z_out, dtype=jnp.float64),
        best_log_score=best_log_score,
        best_cell_index=jnp.asarray(best_cell_index, dtype=jnp.int64),
        max_posterior=max_posterior,
        kernel_row_image_ids=kernel_row_image_ids,
        row_is_valid=row_is_valid,
        classes=classes,
    )


def _class_sub_segment_posterior(scores, rows, row_is_valid, *, n_segments: int, n_classes: int):
    """Each (slot, class) sub-segment's log-Z and first maximum, for the per-class statistics.

    The scores are the joint-min-centred ones the image's posterior normalizes,
    so a class's log-Z plus the image's offset is its absolute evidence and
    ``best - logZ_image`` its share of Pmax. The log-Z is the segmented kernel
    the image posterior uses, run on the sub-segments; the winner is the first
    maximum in row order, as for the image (:func:`_winner_take_all_cells`).
    """

    from relax.cuda import posterior as cuda_posterior

    layout = rows.classes
    log_z = cuda_posterior.sparse_pass2_segmented_log_z_f64(
        scores.reshape(-1),
        layout.segment_offsets,
        rows.n_valid_images * jnp.int32(n_classes),
    )
    best_log_score, best_cell_index, _winner = _winner_take_all_cells(
        scores,
        layout.row_segment,
        row_is_valid,
        layout.segment_offsets,
        image_capacity=n_segments,
    )
    return resident_chunk_records.ChunkClassPosterior(
        log_z=jnp.asarray(log_z, dtype=jnp.float64),
        best_log_score=best_log_score,
        best_cell_index=best_cell_index,
    )


def _winner_take_all_cells(scores, row_image_local, row_is_valid, segment_offsets, *, image_capacity: int):
    """Each image's first maximum over its segment, and the one-hot posterior on it.

    ``scores`` is ``[C_R, T]`` with ``-inf`` on invalid cells; rows are
    image-major, so a flat cell's segment-relative index is its offset from the
    image's first cell. Ties go to the smallest flat cell, as ``jnp.argmax``
    does over the compact engine's bucket rows in the same order. Returns
    ``(best_log_score [C_B], best_cell_index int64 [C_B], posterior [C_R, T])``;
    an image without a finite score has ``best_log_score = -inf``, cell 0 and
    no posterior mass.
    """

    row_capacity, n_fine_trans = scores.shape
    row_image = jnp.where(row_is_valid, row_image_local, jnp.int32(image_capacity))
    best_log_score = jax.ops.segment_max(
        jnp.max(scores, axis=1), row_image, num_segments=image_capacity + 1, indices_are_sorted=True
    )[:image_capacity]
    cell = jnp.arange(row_capacity * n_fine_trans, dtype=jnp.int64).reshape(row_capacity, n_fine_trans)
    row_best = best_log_score[jnp.minimum(row_image, image_capacity - 1)]
    is_best = row_is_valid[:, None] & jnp.isfinite(scores) & (scores == row_best[:, None])
    first_cell = jnp.min(jnp.where(is_best, cell, jnp.iinfo(jnp.int64).max), axis=1)
    best_flat = jax.ops.segment_min(
        first_cell, row_image, num_segments=image_capacity + 1, indices_are_sorted=True
    )[:image_capacity]
    has_winner = jnp.isfinite(best_log_score)
    segment_start = jnp.asarray(segment_offsets[:image_capacity], dtype=jnp.int64)
    best_cell_index = jnp.where(has_winner, best_flat - segment_start, jnp.int64(0))
    posterior = jnp.zeros((row_capacity * n_fine_trans,), dtype=jnp.float32).at[
        jnp.where(has_winner, best_flat, row_capacity * n_fine_trans)
    ].set(1.0, mode="drop")
    return best_log_score, best_cell_index, posterior.reshape(row_capacity, n_fine_trans)


def _resident_chunk_posterior_firstiter_cc(
    rows: resident_chunk_records.ChunkRowArrays,
    operands: resident_chunk_records.ChunkStageOperands,
    tables: resident_chunk_records.ChunkStageTables,
    *,
    spec: resident_chunk_records.ChunkProgramSpec,
    row_is_valid,
    kernel_row_image_ids,
) -> resident_chunk_records.ChunkPosterior:
    """RELION's ``--firstiter_cc`` iteration: normalized-CC scores, winner takes all.

    RELION zeroes every weight but the best one (ml_optimiser.cpp:9266-9293;
    acc_ml_optimiser_impl.h:2868-2960), so the posterior is one-hot at each
    image's highest CC and Pmax is 1. The winner is the first maximum in the
    image's segment order, which is the compact engine's ``jnp.argmax`` over its
    bucket rows in the same order (``_winner_take_all_bucket_probs``). log-Z is
    the log-sum-exp of the CC scores, the compact engine's reported evidence.
    """

    from relax.fine_pass.resident_scoring import score_resident_chunk_normalized_cc

    scored = score_resident_chunk_normalized_cc(
        rows.row_image_local,
        rows.row_fine_rot,
        rows.row_mask_bits,
        rows.row_mask_mode,
        rows.n_valid_rows,
        tables.projection_score_cache,
        operands.score_shifted_cc,
        operands.corr_img_score,
        operands.cc_half_batch_norm,
        half_weights=tables.half_weights,
        full_to_compact=tables.full_to_compact,
        fine_translation_parent=tables.fine_translation_parent,
        row_capacity=spec.row_capacity,
        n_fine_trans=spec.n_fine_trans,
        block_rows=int(spec.mstep_block_rows),
    )
    return _winner_take_all_posterior(
        scored,
        rows,
        image_capacity=spec.image_capacity,
        row_is_valid=row_is_valid,
        kernel_row_image_ids=kernel_row_image_ids,
    )


def _winner_take_all_posterior(
    scored, rows, *, image_capacity: int, row_is_valid, kernel_row_image_ids
) -> resident_chunk_records.ChunkPosterior:
    """The ``--firstiter_cc`` posterior of a chunk's normalized-CC scores (one segment per image or particle)."""

    from relax.cuda import posterior as cuda_posterior

    scores = jnp.asarray(scored.scores, dtype=jnp.float32)
    scores_flat = scores.reshape(-1)
    log_z = cuda_posterior.sparse_pass2_segmented_log_z_f64(
        scores_flat, rows.segment_offsets, rows.n_valid_images
    )

    best_log_score, best_cell_index, winner = _winner_take_all_cells(
        scores,
        rows.row_image_local,
        row_is_valid,
        rows.segment_offsets,
        image_capacity=image_capacity,
    )
    has_winner = jnp.isfinite(best_log_score)
    return resident_chunk_records.ChunkPosterior(
        row_posterior=winner,
        min_diff2=scored.min_diff2,
        class_log_z=jnp.asarray(log_z, dtype=jnp.float64),
        best_log_score=best_log_score,
        best_cell_index=best_cell_index,
        max_posterior=has_winner.astype(jnp.float32),
        kernel_row_image_ids=kernel_row_image_ids,
        row_is_valid=row_is_valid,
    )


def _cached_block_projections(tables: resident_chunk_records.ChunkStageTables, block_fine_rot):
    """The global pass's block projections: three gathers by fine-rotation id.

    Split out so the M-step block body takes its projections as operands. The
    global pass keeps gathering them out of the per-iteration caches exactly
    where it did before; local search (T12) has no cacheable fine grid and
    slices the projections it computed for the chunk's own rows instead.
    """

    if tables.union_recon_take is not None:
        # The union cache: the reconstruction window is taken out of the
        # gathered rows and |recon|^2 is formed as the three-cache build formed
        # it (_place_windowed_projection_block).
        recon = cache_rows(tables.projection_score_cache, block_fine_rot)[:, tables.union_recon_take]
        recon_abs2 = (jnp.abs(recon) ** 2).astype(jnp.real(recon).dtype)
        if tables.residual_sgd_take is None:
            return recon, recon_abs2, tables.mstep_grid[block_fine_rot]
        # VDAM's subtracted reference: the SGD kernel's projection, which keeps
        # the rows the fine and wavg kernels zero (_sgd_residual_rows).
        residual = recon.at[:, tables.residual_sgd_take].set(
            cache_rows(tables.residual_sgd_cache, block_fine_rot).astype(recon.dtype)
        )
        return recon, recon_abs2, tables.mstep_grid[block_fine_rot], residual
    return (
        tables.projection_recon_cache[block_fine_rot],
        tables.projection_recon_abs2_cache[block_fine_rot],
        tables.mstep_grid[block_fine_rot],
    )


def _resident_mstep_block(
    *,
    block_row_image,
    block_kernel_ids,
    block_posterior,
    block_projections,
    block_sum_ids=None,
    operands: resident_chunk_records.ChunkStageOperands,
    tables: resident_chunk_records.ChunkStageTables,
    carry: resident_chunk_records.ChunkMstepCarry,
    spec: resident_chunk_records.ChunkProgramSpec,
) -> resident_chunk_records.ChunkMstepCarry:
    """One pixel-axis row block: weighted sums, Wavg, noise, both adjoints.

    Statement for statement the body the per-stage loop ran inline; every path
    calls this one copy, so the only differences between them are how the
    block's rows are sliced (static Python slice versus ``dynamic_slice``) and
    where its projections come from. ``block_projections`` is the block's
    ``(projection, |projection|^2, M-step rotations)``, with VDAM's residual
    projection as a fourth entry when it differs (``_sgd_residual_rows``): the
    global pass passes ``_cached_block_projections(tables, block_fine_rot)``,
    local search passes a slice of the projections it computed for this chunk.
    """

    from relax.cuda import fine as cuda_fine

    proj, proj_abs2, block_mstep_rotations = block_projections[:3]
    # The SGD residual's projection when it differs from the Wavg one (_cached_block_projections).
    residual_proj = block_projections[3] if len(block_projections) > 3 else proj
    if tables.window_logical is None:
        logical_recon_pixels = jnp.asarray(spec.n_recon_pixels, dtype=jnp.int32)
        logical_rect_pixels = jnp.asarray(spec.n_rect, dtype=jnp.int32)
    else:
        logical_recon_pixels = tables.window_logical.recon_pixels
        logical_rect_pixels = tables.window_logical.rect_pixels
    block_optics_groups = (
        None if operands.optics_groups is None else operands.optics_groups[block_row_image]
    )

    if spec.use_translate_sum_kernel:
        summed, summed_masked, ctf_probs, _probs_sum_t = resident_block_programs.resident_block_weighted_sums_kernel(
            block_posterior,
            block_kernel_ids,
            block_row_image,
            operands.recon_image,
            operands.recon_weight,
            operands.noise_image,
            operands.ctf2_over_nv_recon,
            tables.recon_pixel_indices,
            tables.translation_angles,
            image_shape=spec.image_shape,
            n_recon_pixels=int(spec.n_recon_pixels),
            kernel_ctf_probs=bool(spec.kernel_ctf_probs),
        )
    else:
        summed, summed_masked, ctf_probs, _probs_sum_t = resident_block_programs.resident_block_weighted_sums(
            block_posterior,
            block_row_image,
            operands.shifted_recon,
            operands.shifted_noise,
            operands.ctf2_over_nv_recon,
        )

    # The backprojected CTF sum. For CTF-premultiplied images its weight is not
    # the noise statistics' CTF^2 / sigma2 (sparse_pass2_bucket_io.premultiplied_bpref_weights).
    bpref_ctf2_over_nv = operands.ctf2_over_nv_recon
    bpref_ctf_probs = ctf_probs
    if operands.bpref_ctf2_over_nv_recon is not None:
        bpref_ctf2_over_nv = operands.bpref_ctf2_over_nv_recon
        bpref_ctf_probs, _ = resident_block_programs.resident_block_ctf_probs(block_posterior, block_row_image, bpref_ctf2_over_nv)

    if spec.stable_window:
        # The physical tail holds real pixels above RELION's cutoff: they add
        # nothing to the weighted sums, the Wavg terms, the noise or either
        # adjoint (the translate-sum kernel's own bound is the physical count).
        recon_live = jnp.arange(int(spec.n_recon_pixels), dtype=jnp.int32) < logical_recon_pixels
        summed = jnp.where(recon_live, summed, jnp.zeros((), summed.dtype))
        summed_masked = jnp.where(recon_live, summed_masked, jnp.zeros((), summed_masked.dtype))
        ctf_probs = jnp.where(recon_live, ctf_probs, jnp.zeros((), ctf_probs.dtype))
        bpref_ctf_probs = jnp.where(recon_live, bpref_ctf_probs, jnp.zeros((), bpref_ctf_probs.dtype))

    if spec.mstep_subtract_ctf_projection:
        summed = resident_block_programs.resident_block_residual(summed, _probs_sum_t, residual_proj, bpref_ctf2_over_nv, block_row_image)

    # RELION Wavg triplet in the flat-row layout, then its rotation atomics.
    # The host tail picks the sequential RELION reducer when the pass carries
    # the RFLOAT CTF operand and the algebraic form otherwise; follow the same
    # branch, resolved on the host into the program's static configuration.
    if not spec.use_rfloat_ctf_wavg:
        exact_terms = resident_block_programs.resident_block_wavg_algebraic_terms(
            proj,
            proj_abs2,
            summed_masked,
            ctf_probs,
            tables.noise_variance_for_noise,
            operands.scale,
            block_row_image,
            block_optics_groups,
        )
    else:
        exact_terms = cuda_fine.relion_wavg_sequential_runtime_flat_rows_triplet_f32(
            jnp.asarray(proj, dtype=jnp.complex64),
            block_kernel_ids,
            jnp.asarray(operands.direct_ctf_rfloat_recon, dtype=jnp.float32),
            jnp.asarray(operands.scale, dtype=jnp.float32),
            jnp.asarray(operands.raw_translated_wavg_for_atomic, dtype=jnp.complex64),
            block_posterior,
            logical_recon_pixels,
        )
    # The image power is added once per chunk after the block walk
    # (_add_chunk_wavg_image_power); the blocks add their rows' exact terms at
    # their rectangle positions, the rest of the rectangle receiving nothing.
    wavg_triplet_pixels = cuda_fine.relion_wavg_exact_atomic_flat_rows_triplet_add_f32(
        jnp.asarray(exact_terms, dtype=jnp.float32),
        block_kernel_ids,
        jnp.asarray(tables.exact_positions, dtype=jnp.int32),
        carry.wavg_triplet_pixels,
        logical_rect_pixels,
    )

    noise_summed_masked, noise_ctf_probs = summed_masked, ctf_probs
    if operands.image_noise_scale is not None:
        # The noise and norm terms are linear in these two sums; the backprojection keeps them whole.
        row_scale = jnp.asarray(operands.image_noise_scale, dtype=jnp.float32)[block_row_image][:, None]
        noise_summed_masked = summed_masked * row_scale.astype(summed_masked.real.dtype)
        noise_ctf_probs = ctf_probs * row_scale.astype(ctf_probs.dtype)
    block_shells, block_a2, block_xa = resident_block_programs.resident_block_noise_and_norm(
        proj,
        proj_abs2,
        noise_summed_masked,
        noise_ctf_probs,
        tables.noise_variance_for_noise,
        tables.shell_indices_noise,
        block_row_image,
        block_optics_groups,
        n_shells=int(spec.stats_config.n_shells),
        image_capacity=int(spec.image_capacity),
        float32_bucketed_image_sums=spec.stats_config.float32_bucketed_image_sums,
    )

    if spec.presum_adjoint:
        # The adjoint is linear in the rows, so a projection's rows are summed
        # here and backprojected once per pass (_backproject_projection_sums).
        # Padded and other-slot rows carry zero sums.
        Ft_y = carry.Ft_y.at[block_sum_ids].add(summed.astype(carry.Ft_y.dtype))
        Ft_ctf = carry.Ft_ctf.at[block_sum_ids].add(bpref_ctf_probs.astype(carry.Ft_ctf.dtype))
        return carry._replace(
            Ft_y=Ft_y,
            Ft_ctf=Ft_ctf,
            wavg_triplet_pixels=wavg_triplet_pixels,
            noise_shells=carry.noise_shells + block_shells,
            a2_per_image=carry.a2_per_image + block_a2,
            xa_per_image=carry.xa_per_image + block_xa,
        )
    Ft_y, Ft_ctf = _backproject_block_rows(
        summed, bpref_ctf_probs, block_mstep_rotations, carry.Ft_y, carry.Ft_ctf, tables=tables, spec=spec
    )
    return carry._replace(
        Ft_y=Ft_y,
        Ft_ctf=Ft_ctf,
        wavg_triplet_pixels=wavg_triplet_pixels,
        noise_shells=carry.noise_shells + block_shells,
        a2_per_image=carry.a2_per_image + block_a2,
        xa_per_image=carry.xa_per_image + block_xa,
    )


def _backproject_block_rows(summed, ctf_probs, block_mstep_rotations, Ft_y, Ft_ctf, *, tables, spec):
    """Both windowed adjoints of one row block: ``summed`` into ``Ft_y``, ``ctf_probs`` into ``Ft_ctf``.

    The block body's adjoint (:func:`_resident_mstep_block`), and the tilt M-step's once it has merged
    each row's translation-block partials (``resident_tilts._tilt_mstep_program``, relax#27).
    """

    runtime_mstep_max_r = None
    if spec.stable_window and tables.window_logical is not None:
        runtime_mstep_max_r = tables.window_logical.mstep_max_r
    adjoint_kwargs = dict(
        window_indices=tables.relion_x_half_recon_indices,
        use_windowed_adjoint=True,
        image_shape=spec.image_shape,
        volume_shape=spec.recon_volume_shape,
        disc_type="linear_interp",
        half_image=True,
        half_volume=True,
        max_r=spec.mstep_max_r,
        relion_x_half=True,
        max_block_bytes=int(spec.max_adjoint_block_bytes),
        runtime_max_r=runtime_mstep_max_r,
    )
    Ft_y = _accumulate_adjoint_block_chunked(
        summed, block_mstep_rotations, Ft_y, log_label="resident-y-window", **adjoint_kwargs
    )
    Ft_ctf = _accumulate_adjoint_block_chunked(
        ctf_probs, block_mstep_rotations, Ft_ctf, log_label="resident-ctf-window", **adjoint_kwargs
    )
    return Ft_y, Ft_ctf


def _mstep_block_operand_dtypes(
    operands: resident_chunk_records.ChunkStageOperands,
    tables: resident_chunk_records.ChunkStageTables,
    *,
    spec: resident_chunk_records.ChunkProgramSpec,
    projection_dtypes=None,
) -> dict:
    """Dtypes of the M-step block's intermediates, by promotion arithmetic.

    Every statement between the block's operands and its three accumulating
    outputs promotes; none of them casts, except the one explicit
    ``astype(float64)`` on the noise shells. So the output dtypes follow from
    the operand dtypes alone, on the host, without tracing anything:

    * ``summed_masked`` is the translate-and-sum kernel's declared complex64
      output, or ``compute_local_weighted_sums`` of the float32 posterior
      against the noise tile;
    * ``ctf_probs`` is the kernel's declared float32 fourth output when it is
      selected, and otherwise ``compute_local_ctf_sums_from_probs_sum_t`` of
      the float32 translation mass against the gathered CTF row;
    * ``A2`` promotes ``|proj|^2``, ``ctf_probs`` and the noise variance;
    * ``XA`` promotes the noise variance against the real part of
      ``proj * conj(summed_masked)``.

    ``_carry_aval_probe_enabled`` checks this against ``jax.eval_shape`` of the
    real block stages; the unit tests set it.
    """

    if projection_dtypes is None:
        # The global pass reads the per-iteration caches; local search has no
        # cache and hands the dtypes of the projections it just computed.
        projection_dtypes = (
            (
                cache_dtype(tables.projection_score_cache),
                jnp.real(jnp.zeros((), dtype=cache_dtype(tables.projection_score_cache))).dtype,
            )
            if tables.union_recon_take is not None
            else (
                tables.projection_recon_cache.dtype,
                tables.projection_recon_abs2_cache.dtype,
            )
        )
    proj_dtype, proj_abs2_dtype = (jnp.dtype(value) for value in projection_dtypes)
    noise_dtype = jnp.dtype(tables.noise_variance_for_noise.dtype)

    if spec.use_translate_sum_kernel:
        summed_masked_dtype = jnp.dtype(jnp.complex64)
    else:
        summed_masked_dtype = jnp.dtype(
            jnp.result_type(jnp.float32, operands.shifted_noise.dtype)
        )
    if spec.use_translate_sum_kernel and spec.kernel_ctf_probs:
        ctf_probs_dtype = jnp.dtype(jnp.float32)
    else:
        ctf_probs_dtype = jnp.dtype(
            jnp.result_type(jnp.float32, operands.ctf2_over_nv_recon.dtype)
        )

    a2_dtype = jnp.dtype(jnp.result_type(proj_abs2_dtype, ctf_probs_dtype, noise_dtype))
    cross_dtype = jnp.dtype(jnp.result_type(proj_dtype, summed_masked_dtype))
    # ``np.zeros`` rather than ``jnp.zeros``: this asks for the real part's
    # dtype, not for a value, and the device version dispatched one
    # ``convert_element_type`` per chunk to allocate a 0-d array that is read
    # for its dtype and thrown away. NumPy's promotion of a real part is the
    # same table JAX consults.
    xa_dtype = jnp.dtype(
        jnp.result_type(noise_dtype, np.zeros((), dtype=cross_dtype).real.dtype)
    )
    return {
        "proj": proj_dtype,
        "proj_abs2": proj_abs2_dtype,
        "summed_masked": summed_masked_dtype,
        "ctf_probs": ctf_probs_dtype,
        "noise": noise_dtype,
        "a2": a2_dtype,
        "xa": xa_dtype,
        # ``_resident_block_noise_and_norm`` casts the binned shells to float64
        # before they leave the block, so the carry is float64 whatever the
        # operands promote to.
        "noise_shells": jnp.dtype(jnp.float64),
    }


def _probe_mstep_block_output_avals(
    tables: resident_chunk_records.ChunkStageTables,
    *,
    spec: resident_chunk_records.ChunkProgramSpec,
    dtypes: dict,
):
    """``jax.eval_shape`` of the block's noise/norm stage, for the aval check.

    This is the probe the chunk loop used to run once per chunk. It is kept as
    a diagnostic only: :func:`_mstep_block_operand_dtypes` is what the driver
    uses, and this function exists so that arithmetic can be proved equal to
    the traced answer.
    """

    block_rows = int(spec.mstep_block_rows)
    n_pixels = int(spec.n_recon_pixels)

    def probe(summed_masked, ctf_probs, proj, proj_abs2, noise, shells, row_image, row_groups):
        return resident_block_programs.resident_block_noise_and_norm(
            proj,
            proj_abs2,
            summed_masked,
            ctf_probs,
            noise,
            shells,
            row_image,
            row_groups,
            n_shells=int(spec.stats_config.n_shells),
            image_capacity=spec.image_capacity,
            float32_bucketed_image_sums=spec.stats_config.float32_bucketed_image_sums,
        )

    return jax.eval_shape(
        probe,
        jax.ShapeDtypeStruct((block_rows, n_pixels), dtypes["summed_masked"]),
        jax.ShapeDtypeStruct((block_rows, n_pixels), dtypes["ctf_probs"]),
        jax.ShapeDtypeStruct((block_rows, n_pixels), dtypes["proj"]),
        jax.ShapeDtypeStruct((block_rows, n_pixels), dtypes["proj_abs2"]),
        tables.noise_variance_for_noise,
        tables.shell_indices_noise,
        jax.ShapeDtypeStruct((block_rows,), jnp.int32),
        (
            jax.ShapeDtypeStruct((block_rows,), jnp.int32)
            if int(spec.stats_config.n_optics_groups) > 1
            else None
        ),
    )


def _check_mstep_carry_avals(
    tables: resident_chunk_records.ChunkStageTables,
    *,
    spec: resident_chunk_records.ChunkProgramSpec,
    dtypes: dict,
) -> None:
    """Raise when the static dtypes disagree with the traced block stages."""

    shells_aval, a2_aval, xa_aval = _probe_mstep_block_output_avals(
        tables, spec=spec, dtypes=dtypes
    )
    expected = (
        (_noise_shell_shape(spec.stats_config), dtypes["noise_shells"]),
        ((int(spec.image_capacity),), dtypes["a2"]),
        ((int(spec.image_capacity),), dtypes["xa"]),
    )
    probed = tuple(
        (tuple(int(size) for size in aval.shape), jnp.dtype(aval.dtype))
        for aval in (shells_aval, a2_aval, xa_aval)
    )
    if probed != expected:
        raise AssertionError(
            "resident M-step carry avals disagree with the traced block stages: "
            f"static={expected} probed={probed}"
        )


@lru_cache(maxsize=None)
def _zero_block_partials(shapes_and_dtypes: tuple) -> Callable[[], tuple]:
    """One program per capacity class that allocates the zero accumulators.

    ``jnp.zeros`` outside a jit is two eager dispatches, a
    ``convert_element_type`` of the scalar zero and a ``broadcast_in_dim`` to
    the shape; the M-step carry has four of them and the driver builds one
    carry per chunk, which on the early state was 3336 eager dispatches over
    two iterations. Inside a program with static shapes and dtypes the same
    four buffers cost none, and each call still returns fresh buffers, which
    the donated M-step block program requires.

    Keyed on the shapes and dtypes, so a class compiles once and every chunk
    of that class reuses it.
    """

    @jax.jit
    def build():
        return tuple(jnp.zeros(shape, dtype=dtype) for shape, dtype in shapes_and_dtypes)

    return build


def _noise_shell_shape(stats_config) -> tuple:
    """``(n_shells,)``, or ``(G, n_shells)`` with G optics groups."""

    groups = int(stats_config.n_optics_groups)
    return ((groups,) if groups > 1 else ()) + (int(stats_config.n_shells),)


def _initial_mstep_carry(
    Ft_y,
    Ft_ctf,
    operands: resident_chunk_records.ChunkStageOperands,
    tables: resident_chunk_records.ChunkStageTables,
    *,
    spec: resident_chunk_records.ChunkProgramSpec,
    projection_dtypes=None,
) -> resident_chunk_records.ChunkMstepCarry:
    """Zero-initialized block accumulators with the block stages' own dtypes.

    The per-image ``A2``/``XA`` partials take whatever dtype the noise and
    projection operands promote to. Those dtypes are computed from the operand
    dtypes and the capacity class by :func:`_mstep_block_operand_dtypes`, which
    is host arithmetic; the chunk loop used to learn them by tracing two
    ``jax.eval_shape`` probes per chunk instead, three traces of Python work
    for an answer that is the same for every chunk of a class.
    Zero-initializing (rather than seeding with the first block, as an earlier
    revision did) makes the loop a ``lax.fori_loop`` carry; adding a leading
    zero changes no float value except the unobservable ``-0.0`` case.
    """

    dtypes = _mstep_block_operand_dtypes(
        operands, tables, spec=spec, projection_dtypes=projection_dtypes
    )
    if policy.carry_aval_probe_enabled():
        _check_mstep_carry_avals(tables, spec=spec, dtypes=dtypes)
    n_class_scale = 2 if int(getattr(spec, "n_classes", 1)) > 1 else 0
    # K>1: the per-image class scale sums are zeroed in the same program.
    wavg_triplet_pixels, noise_shells, a2_per_image, xa_per_image, *scale_sums = _zero_block_partials(
        (
            ((spec.image_capacity, int(spec.n_rect), 3), jnp.dtype(jnp.float32)),
            (_noise_shell_shape(spec.stats_config), jnp.dtype(dtypes["noise_shells"])),
            ((spec.image_capacity,), jnp.dtype(dtypes["a2"])),
            ((spec.image_capacity,), jnp.dtype(dtypes["xa"])),
        )
        + (((spec.image_capacity,), jnp.dtype(jnp.float64)),) * n_class_scale
    )()
    class_scale = {}
    if n_class_scale:
        class_scale = dict(scale_xa_per_image=scale_sums[0], scale_aa_per_image=scale_sums[1])
    return resident_chunk_records.ChunkMstepCarry(
        Ft_y=Ft_y,
        Ft_ctf=Ft_ctf,
        wavg_triplet_pixels=wavg_triplet_pixels,
        noise_shells=noise_shells,
        a2_per_image=a2_per_image,
        xa_per_image=xa_per_image,
        **class_scale,
    )


def _resident_mstep_block_at(
    block_start,
    blocks: resident_chunk_records.MstepBlockInputs,
    operands: resident_chunk_records.ChunkStageOperands,
    tables: resident_chunk_records.ChunkStageTables,
    carry: resident_chunk_records.ChunkMstepCarry,
    *,
    spec: resident_chunk_records.ChunkProgramSpec,
) -> resident_chunk_records.ChunkMstepCarry:
    """Slice one block out of the chunk's row arrays and run the block body.

    The per-stage loop did these four slices and three gathers as loose eager
    operations, one dispatch each per block. They are the same slices: every
    row capacity is a whole number of blocks, so ``block_start + block_rows``
    never exceeds the capacity and ``dynamic_slice_in_dim`` never clamps, which
    makes the rows it returns the rows the Python slice returned.
    """

    block_rows = int(spec.mstep_block_rows)

    def take(values):
        return jax.lax.dynamic_slice_in_dim(values, block_start, block_rows, axis=0)

    block_sum_ids = None
    if blocks.projections is None:
        block_row_ids = take(blocks.row_fine_rot)
        block_projections = _cached_block_projections(tables, block_row_ids)
        if spec.presum_adjoint:
            # The row's fine rotation within its class: its projection id, or
            # a class's (or a streamed chunk's) slot mapped back through the table.
            block_sum_ids = (
                block_row_ids
                if tables.cache_slot_fine_rot is None
                else jnp.asarray(tables.cache_slot_fine_rot, dtype=jnp.int32)[block_row_ids]
            )
    else:
        block_projections = blocks.projections
    if spec.presum_adjoint and blocks.row_sum_ids is not None:
        block_sum_ids = take(blocks.row_sum_ids)
    block_kernel_ids = take(blocks.kernel_row_image_ids)
    block_posterior = take(blocks.row_posterior)
    if blocks.class_row_range is not None:
        # A boundary block's rows of the neighbouring class are treated as
        # padding (no weight, kernel id -1). A block past the capacity is
        # clamped by the slice, but its unclamped positions are all at or past
        # ``hi``, so every row of it is excluded.
        row = block_start + jnp.arange(block_rows, dtype=jnp.int32)
        in_class = (row >= blocks.class_row_range[0]) & (row < blocks.class_row_range[1])
        block_kernel_ids = jnp.where(in_class, block_kernel_ids, jnp.int32(-1))
        block_posterior = jnp.where(in_class[:, None], block_posterior, jnp.zeros((), block_posterior.dtype))
    return _resident_mstep_block(
        block_row_image=take(blocks.row_image_local),
        block_kernel_ids=block_kernel_ids,
        block_posterior=block_posterior,
        block_projections=block_projections,
        block_sum_ids=block_sum_ids,
        operands=operands,
        tables=tables,
        carry=carry,
        spec=spec,
    )


@partial(jax.jit, static_argnames=("spec",), donate_argnums=(4,))
def _resident_mstep_block_program(
    block_start,
    blocks: resident_chunk_records.MstepBlockInputs,
    operands: resident_chunk_records.ChunkStageOperands,
    tables: resident_chunk_records.ChunkStageTables,
    carry: resident_chunk_records.ChunkMstepCarry,
    *,
    spec: resident_chunk_records.ChunkProgramSpec,
) -> resident_chunk_records.ChunkMstepCarry:
    """One M-step block as one program, keyed on the capacity and pixel class.

    Same statements, same order, same dtypes as the loose dispatch; the only
    change is where the JIT boundary sits. The carry is donated so the two
    half-volumes the windowed adjoint accumulates into keep being updated in
    place, as they are when the adjoint FFI is dispatched on its own.
    """

    return _resident_mstep_block_at(
        block_start,
        blocks,
        operands,
        tables,
        carry,
        spec=spec,
    )


@partial(jax.jit, static_argnames=("spec",))
def _resident_chunk_posterior_program(
    rows: resident_chunk_records.ChunkRowArrays,
    operands: resident_chunk_records.ChunkStageOperands,
    tables: resident_chunk_records.ChunkStageTables,
    *,
    spec: resident_chunk_records.ChunkProgramSpec,
) -> resident_chunk_records.ChunkPosterior:
    """:func:`_resident_chunk_posterior` as one program per capacity class.

    The stage's own arithmetic is unchanged; what leaves the chunk loop is the
    dozen eager operations around it -- the two ``arange`` row/image masks, the
    logical current size, the score reshape and the posterior's unit external
    weight -- each of which was a dispatch and a single-primitive program.
    """

    return _resident_chunk_posterior(rows, operands, tables, spec=spec)


@partial(jax.jit, static_argnames=("spec",), donate_argnums=(0,))
def _resident_chunk_statistics_program(
    stats,
    rows: resident_chunk_records.ChunkRowArrays,
    operands: resident_chunk_records.ChunkStageOperands,
    tables: resident_chunk_records.ChunkStageTables,
    posterior: resident_chunk_records.ChunkPosterior,
    mstep: resident_chunk_records.ChunkMstepCarry,
    *,
    spec: resident_chunk_records.ChunkProgramSpec,
):
    """The chunk's Wavg image power, then :func:`_resident_chunk_statistics`, as one program per capacity class.

    The image power (:func:`_add_chunk_wavg_image_power`) is read only by the
    statistics, so it runs inside this program rather than as one of its own.
    The statistics accumulator is donated: it is a running total the driver
    rebinds every chunk, so updating it in place is what the loose dispatch
    already did through ``_accumulate_chunk_image_terms``.
    """

    mstep = resident_block_programs.add_chunk_wavg_image_power(
        mstep, operands, tables, posterior.row_posterior, posterior.kernel_row_image_ids, spec=spec
    )
    return _resident_chunk_statistics(
        stats, rows, operands, tables, posterior, mstep, spec=spec
    )


def _resident_chunk_statistics(
    stats,
    rows: resident_chunk_records.ChunkRowArrays,
    operands: resident_chunk_records.ChunkStageOperands,
    tables: resident_chunk_records.ChunkStageTables,
    posterior: resident_chunk_records.ChunkPosterior,
    mstep: resident_chunk_records.ChunkMstepCarry,
    *,
    spec: resident_chunk_records.ChunkProgramSpec,
):
    """Fold one chunk's image-level terms and its padding sanity counter."""

    image_tables = resident_image_terms.ChunkImageTables(
        shell_indices_half=tables.shell_indices_half,
        wavg_shell_indices=tables.wavg_shell_indices,
        wavg_scale_pixel_mask=tables.wavg_scale_pixel_mask,
        translation_sqdist_ang=operands.translation_sqdist_ang,
        norm_shell_cutoff=(
            None
            if tables.window_logical is None
            else (
                tables.window_logical.current_size
                if tables.window_logical.wsum_current_size is None
                else tables.window_logical.wsum_current_size
            )
            // 2
        ),
    )
    best_row_local = posterior.best_cell_index // jnp.int64(int(spec.n_fine_trans))
    slot_is_valid = jnp.arange(int(spec.image_capacity), dtype=jnp.int32) < rows.n_valid_images
    invalid_best = slot_is_valid & (
        (best_row_local < 0) | (best_row_local >= rows.image_row_count)
    )
    best_chunk_row = jnp.clip(
        rows.image_row_start + best_row_local,
        0,
        jnp.int64(max(int(spec.row_capacity) - 1, 0)),
    ).astype(jnp.int32)
    best_fine_rot = jnp.asarray(rows.row_fine_rot, dtype=jnp.int64)[best_chunk_row]
    if tables.cache_slot_fine_rot is not None:
        best_fine_rot = jnp.asarray(tables.cache_slot_fine_rot, dtype=jnp.int64)[best_fine_rot]

    class_fields = {}
    if posterior.classes is not None:
        # Each class's own winner: its sub-segment row, then that row's fine
        # rotation through the class-stacked (or streamed) slot table.
        n_fine_trans = jnp.int64(int(spec.n_fine_trans))
        class_best_row = jnp.clip(
            rows.classes.segment_row_start + posterior.classes.best_cell_index // n_fine_trans,
            0,
            jnp.int64(max(int(spec.row_capacity) - 1, 0)),
        ).astype(jnp.int32)
        class_fine_rot = jnp.asarray(tables.cache_slot_fine_rot, dtype=jnp.int64)[
            jnp.asarray(rows.row_fine_rot, dtype=jnp.int64)[class_best_row]
        ]
        class_fields = dict(
            row_class=rows.classes.row_class,
            per_class_log_z=posterior.classes.log_z,
            per_class_best_log_score=posterior.classes.best_log_score,
            per_class_best_cell=jnp.where(
                jnp.isfinite(posterior.classes.best_log_score),
                class_fine_rot * n_fine_trans + posterior.classes.best_cell_index % n_fine_trans,
                jnp.int64(-1),
            ),
        )

    chunk_operands = resident_image_terms.ChunkImageOperands(
        row_posterior=posterior.row_posterior,
        row_image_local=rows.row_image_local,
        row_coarse_rot=jnp.where(
            posterior.row_is_valid,
            tables.coarse_parent_grid[rows.row_fine_rot],
            jnp.int32(int(spec.stats_config.n_coarse_rot)),
        ),
        image_ids=rows.image_ids,
        group_ids=operands.group_ids,
        image_power_shells=operands.image_power_shells,
        relion_norm_high_shell=operands.relion_norm_high_shell,
        wavg_triplet_pixels=mstep.wavg_triplet_pixels,
        block_noise_shells=mstep.noise_shells,
        a2_per_image=mstep.a2_per_image,
        xa_per_image=mstep.xa_per_image,
        class_log_z=posterior.class_log_z,
        min_diff2=posterior.min_diff2,
        best_log_score=posterior.best_log_score,
        max_posterior=posterior.max_posterior,
        best_cell_index=posterior.best_cell_index,
        best_fine_rot=best_fine_rot,
        optics_groups=operands.optics_groups,
        scale_xa_per_image=mstep.scale_xa_per_image,
        scale_aa_per_image=mstep.scale_aa_per_image,
        **class_fields,
    )
    stats = resident_image_terms.accumulate_chunk_image_terms(
        stats, chunk_operands, image_tables, config=spec.stats_config
    )
    return stats._replace(
        invalid_best_rows=stats.invalid_best_rows + jnp.sum(invalid_best.astype(jnp.int64))
    )


# Divisors of the M-step block for classes with few live rows. The pixel-axis
# block costs in proportion to its rows whether or not they carry weight, and
# at 50k/256 one class of a chunk has about 40 live rows against a 2048-row
# block. Three sizes bound the programs a capacity class compiles.
_LIVE_BLOCK_DIVISORS = (64, 8, 1)
_MIN_LIVE_BLOCK_ROWS = 16


def _live_block_spec(spec: "resident_chunk_records.ChunkProgramSpec", n_live_rows: int) -> "resident_chunk_records.ChunkProgramSpec":
    """The chunk spec whose M-step block is the smallest ladder size holding ``n_live_rows``.

    The ladder is the configured block divided by 64, by 8 and by 1 (no smaller
    than 16 rows); a slot with more live rows than the configured block walks
    it in configured blocks as before. Every ladder size is a power of two that
    divides the row capacity, so a block never runs past the capacity.
    """

    full = int(spec.mstep_block_rows)
    for divisor in _LIVE_BLOCK_DIVISORS:
        rows = max(full // divisor, min(_MIN_LIVE_BLOCK_ROWS, full))
        if int(n_live_rows) <= rows:
            break
    return spec if rows == full else dataclass_replace(spec, mstep_block_rows=rows)


def _run_resident_chunk_stages(
    rows: resident_chunk_records.ChunkRowArrays,
    operands: resident_chunk_records.ChunkStageOperands,
    tables: resident_chunk_records.ChunkStageTables,
    carry: tuple,
    *,
    spec: resident_chunk_records.ChunkProgramSpec,
    timing_hook=None,
    row_partial: RowPartial | None = None,
):
    """The chunk's stages, dispatched one at a time.

    The host block loop reads each accumulator slot's live row range back from
    the device once per chunk (:func:`_make_mstep_block_inputs`), so it
    launches only the blocks that carry weight.

    ``timing_hook(name)`` is called after each stage when the chunk
    timing diagnostic is on; it synchronizes, so an arm that passes it is a
    diagnostic arm.
    """

    front = _resident_chunk_stages_front(rows, operands, tables, spec=spec, timing_hook=timing_hook)
    return _resident_chunk_stages_finish(
        front, rows, operands, tables, carry, spec=spec, timing_hook=timing_hook, row_partial=row_partial
    )


def _resident_chunk_stages_front(rows, operands, tables, *, spec, timing_hook=None):
    """The per-stage path's posterior and M-step row order, enqueued; no host read."""

    if policy.resident_glue_jit_enabled():
        posterior = _resident_chunk_posterior_program(rows, operands, tables, spec=spec)
    else:
        posterior = _resident_chunk_posterior(rows, operands, tables, spec=spec)
    if timing_hook is not None:
        timing_hook("posterior", posterior.row_posterior)
    return posterior, _make_mstep_block_inputs(rows, posterior, n_slots=int(spec.n_slots))


def _resident_chunk_stages_finish(
    front,
    rows,
    operands,
    tables,
    carry,
    *,
    spec,
    timing_hook=None,
    block_projections=None,
    projection_dtypes=None,
    row_partial: RowPartial | None = None,
):
    """The per-stage path after its front: the live row ranges, the M-step blocks, the statistics.

    ``block_projections(start, block_rows)``, when given, returns the M-step
    block's ``(proj, |proj|^2, M-step rotations)`` for the block at ``start`` of
    the M-step order, instead of the block gathering them out of the tables'
    caches: a lone overflow chunk of a streamed pass projects each live block's
    rows (:func:`_run_lone_resident_chunk`); ``projection_dtypes`` are then
    the ``(proj, |proj|^2)`` dtypes it returns, which the tables' caches
    (absent there) would otherwise give.

    With ``row_partial`` the blocks scatter into the pass's partial pair, which is added into the slot's
    totals every ``blocks_per_flush`` blocks and when the slot's blocks end (relax#65).
    """

    glue_jit = policy.resident_glue_jit_enabled()
    posterior, blocks = front
    Ft_y_total, Ft_ctf_total, stats = carry
    mstep = _initial_mstep_carry(
        Ft_y_total[0], Ft_ctf_total[0], operands, tables, spec=spec, projection_dtypes=projection_dtypes
    )
    slot_offsets = np.asarray(jax.device_get(blocks.slot_offsets), dtype=np.int64)
    Ft_y_out, Ft_ctf_out = [], []
    for slot_index in range(int(spec.n_slots)):
        row_lo, row_hi = int(slot_offsets[slot_index]), int(slot_offsets[slot_index + 1])
        slot_y, slot_ctf = Ft_y_total[slot_index], Ft_ctf_total[slot_index]
        mstep = mstep._replace(Ft_y=slot_y, Ft_ctf=slot_ctf) if row_partial is None else row_partial.bind(mstep)
        class_blocks = blocks._replace(class_row_range=blocks.slot_offsets[slot_index : slot_index + 2])
        block_spec = _live_block_spec(spec, row_hi - row_lo)
        block_rows = int(block_spec.mstep_block_rows)
        # Blocks past the slot's live rows hold only rows without weight, of
        # this slot or another: the weighted sums, the Wavg terms, the noise
        # partials and both adjoint scatters would add exact zeros.
        for start in range((row_lo // block_rows) * block_rows, row_hi, block_rows):
            start_blocks = (
                class_blocks
                if block_projections is None
                else class_blocks._replace(row_fine_rot=None, projections=block_projections(start, block_rows))
            )
            if glue_jit:
                mstep = _resident_mstep_block_program(
                    resident_chunk_records.device_int32(start), start_blocks, operands, tables, mstep, spec=block_spec
                )
            else:
                mstep = _resident_mstep_block_at(
                    resident_chunk_records.device_int32(start),
                    start_blocks,
                    operands,
                    tables,
                    mstep,
                    spec=block_spec,
                )
            if row_partial is not None:
                mstep, slot_y, slot_ctf = row_partial.block_done(mstep, slot_y, slot_ctf, block_rows=block_rows)
        if row_partial is not None:
            mstep, slot_y, slot_ctf = row_partial.flush(mstep, slot_y, slot_ctf)
        else:
            slot_y, slot_ctf = mstep.Ft_y, mstep.Ft_ctf
        if mstep.scale_xa_per_image is not None:
            # Slot ``class + K * group``: the scale sums are masked by the slot's class.
            mstep = _fold_class_scale_sums(
                mstep, tables.wavg_scale_pixel_mask, resident_chunk_records.device_int32(slot_index % int(spec.n_classes))
            )
        Ft_y_out.append(slot_y)
        Ft_ctf_out.append(slot_ctf)
    if timing_hook is not None:
        timing_hook("mstep", (Ft_y_out, Ft_ctf_out))

    if glue_jit:
        # The program adds the chunk's Wavg image power first, as below.
        stats = _resident_chunk_statistics_program(
            stats, rows, operands, tables, posterior, mstep, spec=spec
        )
    else:
        mstep = resident_block_programs.add_chunk_wavg_image_power(
            mstep, operands, tables, posterior.row_posterior, posterior.kernel_row_image_ids, spec=spec
        )
        stats = _resident_chunk_statistics(
            stats, rows, operands, tables, posterior, mstep, spec=spec
        )
    return tuple(Ft_y_out), tuple(Ft_ctf_out), stats


def _run_resident_chunk(
    chunk,
    *,
    tables,
    window_logical=None,
    experiment_dataset,
    bucket_io_kwargs,
    half_weights,
    translation_angles,
    full_to_compact,
    n_score_pixels,
    fine_translation_prior_2d,
    score_real_dtype,
    projection_score_cache,
    projection_recon_cache,
    projection_recon_abs2_cache,
    fine_translation_parent_device,
    mstep_grid,
    coarse_parent_grid,
    n_fine_trans,
    n_recon_windowed,
    n_rect,
    mstep_block_rows,
    adaptive_fraction,
    windowed_prepare,
    stream_projection_fn=None,
    n_fine_rot=None,
    cache_slot_fine_rot=None,
    slot_of_projection=None,
    window_indices,
    recon_window_indices,
    relion_x_half_recon_indices,
    exact_positions_device,
    rect_indices_device,
    recon_pixel_indices,
    resident_operands,
    verify_operands,
    image_shape,
    current_size,
    mstep_current_size,
    wsum_current_size=None,
    chunk_unshifted_operands=False,
    precision_policy=None,
    mstep_max_r,
    recon_volume_shape,
    max_adjoint_block_bytes,
    noise_variance_for_noise,
    program_current_size=None,
    shell_indices_noise,
    group_ids_np,
    scale_corrections_np,
    translation_prior_centers_np,
    fine_translations,
    voxel_size,
    use_exact_relion_gaussian,
    accumulate_noise,
    source_faithful_spectrum_norm,
    stats,
    stats_config,
    image_tables,
    Ft_y_total,
    Ft_ctf_total,
    submitted_keys=None,
    optics_groups_np=None,
    relion_native_fine_units=False,
    coarse_reuse=None,
    firstiter_cc=False,
    mstep_subtract_ctf_projection=False,
    union_score_take=None,
    union_recon_take=None,
    union_native_fft_size=0,
    residual_sgd_cache=None,
    residual_sgd_take=None,
    deferred=False,
    lone_block_rows=None,
    row_partial: RowPartial | None = None,
):
    """Run every resident stage for one capacity chunk.

    ``row_partial`` is the pass's partial accumulator pair, when its rows are backprojected one by one
    and the pair fits the size rule (``half_volume_mstep.allocate_row_partial``).

    ``lone_block_rows`` is set for one image's overflow chunk (a chunk past the
    plan's largest row class, :func:`chunk_runs_alone`), which runs alone and is
    scored and reconstructed in blocks of that many rows
    (:func:`_run_lone_resident_chunk`, :func:`lone_block_rows`).

    ``union_score_take`` / ``union_recon_take`` / ``union_native_fft_size``
    describe a union projection cache (see ``_ChunkStageTables``); they are
    None / 0 for the three-cache layout and for streamed chunks.
    ``residual_sgd_cache`` / ``residual_sgd_take`` are VDAM's subtracted-reference
    rows of that cache (``_sgd_residual_rows``), None when it has none.

    ``submitted_keys``, when given, collects the ``(program name, spec)`` keys
    this chunk submits, so the driver can say how many of them the compile-ahead
    warm-up had already compiled. Recording costs a set insert per chunk.

    Returns the updated ``(Ft_y_total, Ft_ctf_total, stats)``. Host work inside
    is the chunk's materialize/pad, its operand preparation and the T7 offsets
    readback the segmented posterior performs internally; no per-chunk result
    is pulled.

    ``deferred`` (per-stage path, timing off) enqueues the chunk up to its
    posterior and M-step row order and returns ``finish(Ft_y_total,
    Ft_ctf_total, stats)`` for the rest, whose live row ranges are read back
    on the host, so the caller can enqueue the next chunk first; the
    accumulator arguments are then unused.
    """

    # The operands' powerClass terms sum above the weighted sums' size (--strict_highres_exp).
    operand_current_size = current_size if wsum_current_size is None else wsum_current_size
    image_indices = np.arange(chunk.image_start, chunk.image_stop, dtype=np.int64)
    timing = policy.chunk_timing_enabled()
    chunk_t0 = time.time()
    stage_t = {}
    if timing:
        jax.block_until_ready(Ft_y_total)
        chunk_t0 = time.time()

    rows = _make_chunk_row_arrays(
        tables, chunk, n_fine_trans, place=resident_chunk_records.PLACE_ON_DEVICE, n_fine_rot=n_fine_rot,
        slot_of_projection=slot_of_projection,
    )
    lone = lone_block_rows is not None
    if stream_projection_fn is not None and not lone:
        (
            rows,
            cache_slot_fine_rot,
            (projection_score_cache, projection_recon_cache, projection_recon_abs2_cache),
            mstep_grid,
            coarse_parent_grid,
        ) = resident_projections.stream_chunk_projections(
            rows,
            _row_projection_ids(_chunk_host_rows(tables, chunk)[1], n_fine_rot if tables.n_classes > 1 else None),
            n_valid_rows=chunk.n_valid_rows,
            row_capacity=chunk.row_capacity,
            project=stream_projection_fn,
            n_fine_rot=n_fine_rot,
            mstep_grid=mstep_grid,
            coarse_parent_grid=coarse_parent_grid,
        )
        if timing:
            jax.block_until_ready(projection_score_cache)
            stage_t["projections"] = time.time() - chunk_t0

    if resident_operands is None and chunk_unshifted_operands:
        recon = unshifted_chunk_operands(
            experiment_dataset,
            image_indices,
            image_capacity=chunk.image_capacity,
            bucket_io_kwargs=bucket_io_kwargs,
            window_indices=window_indices,
            recon_window_indices=recon_window_indices,
            rect_indices_device=rect_indices_device,
            exact_positions_device=exact_positions_device,
            translation_angles=translation_angles,
            noise_shell_indices_half=image_tables.shell_indices_half,
            n_noise_shells=int(stats_config.n_shells),
            image_shape=image_shape,
            current_size=operand_current_size,
            n_fine_trans=int(n_fine_trans),
            use_exact_relion_gaussian=use_exact_relion_gaussian,
            accumulate_noise=accumulate_noise,
            source_faithful_spectrum_norm=bool(source_faithful_spectrum_norm),
            fine_translation_prior_2d=fine_translation_prior_2d,
            scale_corrections_np=scale_corrections_np,
            group_ids_np=group_ids_np,
            optics_groups_np=optics_groups_np,
            precision_policy=precision_policy,
            relion_native_fine_units=relion_native_fine_units,
        )
    elif resident_operands is None:
        recon = _prepare_chunk_reconstruction_operands(
            chunk=chunk,
            image_indices=image_indices,
            experiment_dataset=experiment_dataset,
            bucket_io_kwargs=bucket_io_kwargs,
            windowed_prepare=windowed_prepare,
            recon_window_indices=recon_window_indices,
            score_window_indices=window_indices,
            fine_translation_prior_2d=fine_translation_prior_2d,
            score_real_dtype=score_real_dtype,
            n_fine_trans=int(n_fine_trans),
            n_recon_windowed=int(n_recon_windowed),
            image_shape=image_shape,
            current_size=operand_current_size,
            use_exact_relion_gaussian=use_exact_relion_gaussian,
            accumulate_noise=accumulate_noise,
            source_faithful_spectrum_norm=source_faithful_spectrum_norm,
            relion_score_translation_angles=translation_angles,
            rect_indices_device=rect_indices_device,
            exact_positions_device=exact_positions_device,
            scale_corrections_np=scale_corrections_np,
            group_ids_np=group_ids_np,
            optics_groups_np=optics_groups_np,
            relion_native_fine_units=relion_native_fine_units,
            normalized_cc=firstiter_cc,
            noise_shell_indices_half=image_tables.shell_indices_half,
            n_noise_shells=int(stats_config.n_shells),
        )
    else:
        # The chunk's image slots are the half's images ``image_start`` to
        # ``image_stop``; the padded slots carry -1 and the gather zeroes them,
        # which is what the per-chunk preparation's capacity mask did.
        image_slots = np.full(chunk.image_capacity, -1, dtype=np.int32)
        image_slots[:chunk.n_valid_images] = image_indices[:chunk.n_valid_images]
        recon = gather_resident_chunk_operands(
            resident_operands,
            image_slots,
            translation_angles=translation_angles,
            rect_indices=rect_indices_device,
            exact_positions=exact_positions_device,
            image_shape=image_shape,
        )
        if verify_operands:
            _verify_resident_chunk_operands(
                recon,
                _prepare_chunk_reconstruction_operands(
                    chunk=chunk,
                    image_indices=image_indices,
                    experiment_dataset=experiment_dataset,
                    bucket_io_kwargs=bucket_io_kwargs,
                    windowed_prepare=windowed_prepare,
                    recon_window_indices=recon_window_indices,
                    score_window_indices=window_indices,
                    fine_translation_prior_2d=fine_translation_prior_2d,
                    score_real_dtype=score_real_dtype,
                    n_fine_trans=int(n_fine_trans),
                    n_recon_windowed=int(n_recon_windowed),
                    image_shape=image_shape,
                    current_size=operand_current_size,
                    use_exact_relion_gaussian=use_exact_relion_gaussian,
                    accumulate_noise=accumulate_noise,
                    source_faithful_spectrum_norm=source_faithful_spectrum_norm,
                    relion_score_translation_angles=translation_angles,
                    rect_indices_device=rect_indices_device,
                    exact_positions_device=exact_positions_device,
                    scale_corrections_np=scale_corrections_np,
                    group_ids_np=group_ids_np,
                    optics_groups_np=optics_groups_np,
                    relion_native_fine_units=relion_native_fine_units,
                    noise_shell_indices_half=image_tables.shell_indices_half,
                    n_noise_shells=int(stats_config.n_shells),
                ),
                translation_angles=translation_angles,
                recon_pixel_indices=recon_pixel_indices,
                image_shape=image_shape,
                n_recon_pixels=int(n_recon_windowed),
                bpref_recon_operand=resident_operands.recon_weight is not None,
                label=f"chunk images {chunk.image_start}-{chunk.image_stop}",
            )
    # The unshifted operands, the half's or this chunk's, carry recon_image; the
    # M-step translates them inside the translate-and-sum kernel.
    unshifted_operands = recon.get("recon_image") is not None
    if timing:
        jax.block_until_ready(recon["recon_image"] if unshifted_operands else recon["shifted_recon"])
        stage_t["operands"] = time.time() - chunk_t0

    # The prior squared distances are a per-image host table; building them at
    # capacity here keeps the program keyed on the capacity class and takes the
    # eager device ops out of the chunk loop. Padded slots multiply a zero
    # posterior, so their value is never observable; they are zeroed anyway.
    translation_sqdist_ang = _make_chunk_translation_sqdist(
        image_tables.translation_sqdist_ang,
        translation_prior_centers_np=translation_prior_centers_np,
        image_indices=image_indices,
        image_capacity=chunk.image_capacity,
        n_valid_images=chunk.n_valid_images,
        fine_translations=fine_translations,
        voxel_size=voxel_size,
    )

    operands = _make_chunk_stage_operands(recon, translation_sqdist_ang)
    stage_tables = _make_chunk_stage_tables(
        projection_score_cache=projection_score_cache,
        projection_recon_cache=projection_recon_cache,
        projection_recon_abs2_cache=projection_recon_abs2_cache,
        mstep_grid=mstep_grid,
        coarse_parent_grid=coarse_parent_grid,
        fine_translation_parent_device=fine_translation_parent_device,
        half_weights=half_weights,
        translation_angles=translation_angles,
        full_to_compact=full_to_compact,
        noise_variance_for_noise=noise_variance_for_noise,
        shell_indices_noise=shell_indices_noise,
        exact_positions_device=exact_positions_device,
        recon_pixel_indices=recon_pixel_indices,
        relion_x_half_recon_indices=relion_x_half_recon_indices,
        image_tables=image_tables,
        cache_slot_fine_rot=cache_slot_fine_rot,
        coarse_reuse=coarse_reuse,
        window_logical=window_logical,
        union_score_take=union_score_take,
        union_recon_take=union_recon_take,
        residual_sgd_cache=residual_sgd_cache,
        residual_sgd_take=residual_sgd_take,
    )
    spec = _make_chunk_program_spec(
        row_capacity=chunk.row_capacity,
        image_capacity=chunk.image_capacity,
        n_fine_trans=n_fine_trans,
        n_score_pixels=n_score_pixels,
        n_recon_pixels=n_recon_windowed,
        n_rect=n_rect,
        mstep_block_rows=mstep_block_rows,
        adaptive_fraction=adaptive_fraction,
        current_size=current_size if program_current_size is None else program_current_size,
        mstep_current_size=mstep_current_size,
        mstep_max_r=mstep_max_r,
        image_shape=image_shape,
        recon_volume_shape=recon_volume_shape,
        max_adjoint_block_bytes=max_adjoint_block_bytes,
        stats_config=stats_config,
        use_rfloat_ctf_wavg=recon["direct_ctf_rfloat_recon"] is not None,
        use_translate_sum_kernel=unshifted_operands,
        bpref_recon_operand=unshifted_operands and recon.get("recon_weight") is not None,
        reuse_coarse_normalization=coarse_reuse is not None,
        firstiter_cc=firstiter_cc,
        n_slots=int(tables.n_slots),
        mstep_subtract_ctf_projection=bool(mstep_subtract_ctf_projection),
        n_classes=tables.n_classes,
        stable_window=program_current_size is not None,
        union_native_fft_size=union_native_fft_size,
        presum_adjoint=resident_projections.carries_projection_sums(Ft_y_total),
    )

    if lone:
        if stream_projection_fn is not None and tables.n_classes > 1:
            # A streamed lone chunk keeps its rows' projection ids, class * n_fine_rot +
            # rotation; the statistics map a winner back to its rotation within the class
            # through this table, as through a streamed chunk's slot table (bench 14694988:
            # Class3D K4 100k crashed on a missing table).
            stage_tables = stage_tables._replace(
                cache_slot_fine_rot=(
                    jnp.arange(tables.n_classes * int(n_fine_rot), dtype=jnp.int32) % jnp.int32(n_fine_rot)
                )
            )
        return _run_lone_resident_chunk(
            rows,
            operands,
            stage_tables,
            spec=spec,
            host_projection_ids=_row_projection_ids(
                _chunk_host_rows(tables, chunk)[1], n_fine_rot if tables.n_classes > 1 else None, slot_of_projection
            ),
            n_valid_rows=chunk.n_valid_rows,
            block_rows=int(lone_block_rows),
            stream_projection_fn=stream_projection_fn,
            deferred=deferred,
            carry=(Ft_y_total, Ft_ctf_total, stats),
            row_partial=row_partial,
        )

    if submitted_keys is not None:
        submitted_keys.update(chunk_program_keys(chunk_program_path(), spec))

    if deferred and not timing:
        front = _resident_chunk_stages_front(rows, operands, stage_tables, spec=spec)

        def finish(Ft_y_total, Ft_ctf_total, stats):
            return _resident_chunk_stages_finish(
                front,
                rows,
                operands,
                stage_tables,
                (Ft_y_total, Ft_ctf_total, stats),
                spec=spec,
                row_partial=row_partial,
            )

        return finish

    def timing_hook(name, value):
        jax.block_until_ready(value)
        stage_t[name] = time.time() - chunk_t0

    Ft_y_total, Ft_ctf_total, stats = _run_resident_chunk_stages(
        rows,
        operands,
        stage_tables,
        (Ft_y_total, Ft_ctf_total, stats),
        spec=spec,
        timing_hook=timing_hook if timing else None,
        row_partial=row_partial,
    )

    if timing:
        jax.block_until_ready((Ft_y_total, Ft_ctf_total, stats.wsum_sigma2_noise))
        total = time.time() - chunk_t0
        operands_s = stage_t.get("operands", 0.0)
        posterior_end = stage_t.get("posterior", operands_s)
        mstep_end = stage_t.get("mstep", total)
        split = "score+posterior=%.3fs mstep=%.3fs statistics=%.3fs" % (
            posterior_end - operands_s,
            mstep_end - posterior_end,
            total - mstep_end,
        )
        blocks = sum(1 for s in range(0, chunk.row_capacity, int(mstep_block_rows)) if s < chunk.n_valid_rows)
        logger.info(
            "Resident pass-2 chunk timing: images=%d/%d rows=%d/%d occupancy=%.3f "
            "mstep_blocks=%d/%d projections=%.3fs operands=%.3fs %s total=%.3fs",
            chunk.n_valid_images, chunk.image_capacity, chunk.n_valid_rows, chunk.row_capacity,
            chunk.n_valid_rows / max(chunk.row_capacity, 1),
            blocks, chunk.row_capacity // int(mstep_block_rows),
            stage_t.get("projections", 0.0),
            operands_s,
            split,
            total,
        )
    return Ft_y_total, Ft_ctf_total, stats


def _run_lone_resident_chunk(
    rows: resident_chunk_records.ChunkRowArrays,
    operands: resident_chunk_records.ChunkStageOperands,
    tables: resident_chunk_records.ChunkStageTables,
    *,
    spec: resident_chunk_records.ChunkProgramSpec,
    host_projection_ids,
    n_valid_rows: int,
    block_rows: int,
    stream_projection_fn,
    deferred: bool,
    carry: tuple,
    row_partial: RowPartial | None = None,
):
    """One image's overflow chunk, scored and reconstructed ``block_rows`` rows at a time.

    An image whose candidates exceed the largest row class (a flat coarse
    posterior keeps every coarse sample, so every fine rotation: 294,912 rows at
    HEALPix 3 in bench 14641043) is one chunk that runs alone. Holding all its
    rows' projections at once does not fit the device, so the rows are scored in
    blocks (:func:`~relax.fine_pass.resident_scoring.score_resident_chunk_in_row_blocks`),
    keeping only their ``[C_R, T]`` diff2; the image's posterior is then formed
    over all of its rows at once, and each live M-step block gathers (cached
    pass) or projects (streamed pass) its own rows' projections. RELION scores
    such an image the same way, in orientation batches against one normalizer.
    The numbers are those of the one-call chunk: only the grouping of the
    projector calls changes.
    """

    from relax.fine_pass.resident_candidates import expand_chunk_mask_jnp
    from relax.fine_pass.resident_scoring import cached_score_reference, score_resident_chunk_in_row_blocks

    if spec.firstiter_cc or (spec.presum_adjoint and stream_projection_fn is not None):
        raise ResidentConfigurationUnsupported(
            "an image past the largest row class runs in row blocks, which the first-iteration "
            "cross-correlation pass and the streamed per-rotation projection sums do not implement"
        )
    host_ids = np.asarray(host_projection_ids, dtype=np.int64)

    def block_ids(ids):
        padded = np.full(block_rows, ids[0] if ids.size else 0, dtype=np.int64)
        padded[: ids.size] = ids
        return padded

    def block_reference(start):
        ids = block_ids(host_ids[start : start + block_rows])
        if stream_projection_fn is not None:
            return stream_projection_fn(ids, block_rows)[0]
        return cached_score_reference(
            tables.projection_score_cache,
            jnp.asarray(ids, dtype=jnp.int32),
            score_take=tables.union_score_take,
            native_fft_size=int(spec.union_native_fft_size),
        )

    row_index = jnp.arange(int(spec.row_capacity), dtype=jnp.int32)
    row_is_valid = row_index < rows.n_valid_rows
    kernel_row_image_ids = jnp.where(row_is_valid, rows.row_image_local, jnp.int32(-1))
    image_index = jnp.arange(int(spec.image_capacity), dtype=jnp.int32)
    scored = score_resident_chunk_in_row_blocks(
        block_reference,
        rows.row_image_local,
        rows.row_log_prior,
        expand_chunk_mask_jnp(rows.row_mask_bits, rows.row_mask_mode, tables.fine_translation_parent),
        int(n_valid_rows),
        jnp.where(image_index < rows.n_valid_images, image_index, jnp.int32(-1)),
        operands.score_input,
        operands.corr_img_score,
        operands.highres_xi2_half,
        operands.translation_prior,
        block_rows=block_rows,
        half_weights=tables.half_weights,
        translation_angles=tables.translation_angles,
        full_to_compact=tables.full_to_compact,
        logical_current_size=_logical_current_size(tables, spec),
        row_capacity=int(spec.row_capacity),
        image_capacity=int(spec.image_capacity),
        n_fine_trans=int(spec.n_fine_trans),
    )
    posterior = _chunk_posterior_from_scores(
        scored,
        rows,
        tables,
        spec=spec,
        row_is_valid=row_is_valid,
        kernel_row_image_ids=kernel_row_image_ids,
    )
    blocks = _make_mstep_block_inputs(rows, posterior, n_slots=int(spec.n_slots))

    block_projections = projection_dtypes = None
    if stream_projection_fn is not None:

        def block_projections(start, n_rows):
            ids = np.asarray(jax.device_get(blocks.row_fine_rot[start : start + n_rows]), dtype=np.int64)
            _, recon, recon_abs2 = stream_projection_fn(block_ids(ids)[:n_rows], n_rows)
            return recon, recon_abs2, tables.mstep_grid[jnp.asarray(ids, dtype=jnp.int32)]

        # A streamed pass has no caches to read the projection dtypes from; one
        # row's projection gives them (the projector call converts on the host,
        # so it cannot be shape-traced).
        _, recon_one, abs2_one = stream_projection_fn(host_ids[:1], 1)
        projection_dtypes = (recon_one.dtype, abs2_one.dtype)

    def finish(Ft_y_total, Ft_ctf_total, stats):
        return _resident_chunk_stages_finish(
            (posterior, blocks),
            rows,
            operands,
            tables,
            (Ft_y_total, Ft_ctf_total, stats),
            spec=spec,
            block_projections=block_projections,
            projection_dtypes=projection_dtypes,
            row_partial=row_partial,
        )

    return finish if deferred else finish(*carry)


def run_resident_mstep_blocks(
    block_projections=None,
    *,
    chunk_projections=None,
    row_capacity: int,
    n_valid_rows: int,
    mstep_block_rows: int,
    image_capacity: int,
    row_image_local,
    kernel_row_image_ids,
    row_posterior,
    recon,
    n_rect: int,
    n_shells: int,
    n_recon_windowed: int,
    noise_variance_for_noise,
    shell_indices_noise,
    exact_positions_device,
    Ft_y_total,
    Ft_ctf_total,
    image_shape,
    recon_volume_shape,
    mstep_current_size,
    mstep_max_r=None,
    relion_x_half_recon_indices,
    max_adjoint_block_bytes,
    n_optics_groups: int = 1,
    row_ids=None,
    recon_pixel_indices=None,
    translation_angles=None,
    row_partial: RowPartial | None = None,
):
    """Walk one chunk's pixel axis in row blocks: Wavg, noise and both adjoints.

    With ``row_partial`` (the pass's partial accumulator pair) the blocks scatter into it, and it is added
    into ``Ft_y_total``/``Ft_ctf_total`` every ``blocks_per_flush`` blocks and after the last (relax#65).

    Exactly one of ``block_projections`` and ``chunk_projections`` is given.

    ``row_ids`` (int32 ``[row_capacity]``) is the chunk row each M-step row
    reads its ``chunk_projections`` at; ``None`` is the identity. The row
    arrays (``row_image_local``, ``kernel_row_image_ids``, ``row_posterior``)
    are already in that order, and ``n_valid_rows`` counts the rows that carry
    weight, so a caller that puts its live rows first launches only their
    blocks.

    ``block_projections(start, stop)`` returns this block's reconstruction-window
    projection, its ``|proj|^2`` and its M-step rotations. The global pass 2
    gathers all three out of the per-iteration fine-rotation caches; local
    search (T12) slices them out of the projections it computed for the chunk's
    own rows, because its fine grid is not cacheable.

    ``chunk_projections`` (P3-G) is the same three arrays for the **whole**
    chunk, in the layout's flat row order. The caller then does no slicing at
    all: the arrays ride in the stage tables and the block program takes a
    ``dynamic_slice`` of the chunk's row ids and reads its rows inside the jit,
    exactly as the global pass reads its per-iteration caches at the block's
    fine-rotation ids. The row ids are ``0 .. row_capacity-1``, so the read is
    the identity gather of the rows the callback sliced, value for value.

    The body is ``_resident_mstep_block``, the same copy the global pass runs in
    both its per-stage and its jitted form, so no accumulator has a second
    implementation. Only the M-step fields of the stage containers are filled
    here: this entry point runs the M-step alone, and the scoring, posterior and
    statistics fields are unused by that body.
    """

    use_translate_sum_kernel = recon.get("recon_image") is not None
    if use_translate_sum_kernel:
        if recon_pixel_indices is None or translation_angles is None:
            raise ValueError(
                "run_resident_mstep_blocks needs recon_pixel_indices and translation_angles "
                "to translate unshifted per-image operands inside the translate-sum kernel"
            )
    elif recon.get("shifted_recon") is None or recon.get("shifted_noise") is None:
        # Fail closed rather than hand ``None`` to the XLA weighted sums.
        raise ValueError(
            "run_resident_mstep_blocks takes either the unshifted per-image operands "
            "('recon_image'/'noise_image', T16) or the per-chunk pre-shifted tiles "
            "('shifted_recon'/'shifted_noise'); this chunk carries neither "
            f"(keys present: {sorted(k for k, v in recon.items() if v is not None)})."
        )

    if (block_projections is None) == (chunk_projections is None):
        raise ValueError(
            "run_resident_mstep_blocks takes exactly one of block_projections "
            "(a callback returning one block's arrays) and chunk_projections "
            "(the chunk's whole row arrays); got "
            f"block_projections={'set' if block_projections is not None else 'None'} "
            f"and chunk_projections={'set' if chunk_projections is not None else 'None'}."
        )
    if chunk_projections is not None:
        chunk_proj, chunk_proj_abs2, chunk_mstep_rotations = chunk_projections
        for name, value in (
            ("projection", chunk_proj),
            ("|projection|^2", chunk_proj_abs2),
            ("M-step rotations", chunk_mstep_rotations),
        ):
            if int(value.shape[0]) != int(row_capacity):
                raise ValueError(
                    "chunk_projections must carry the chunk's whole row axis: "
                    f"the {name} array has {int(value.shape[0])} rows, the chunk "
                    f"capacity is {int(row_capacity)}."
                )

    spec = resident_chunk_records.ChunkProgramSpec(
        row_capacity=int(row_capacity),
        image_capacity=int(image_capacity),
        n_fine_trans=int(row_posterior.shape[1]),
        n_score_pixels=0,
        n_recon_pixels=int(n_recon_windowed),
        n_rect=int(n_rect),
        mstep_block_rows=int(mstep_block_rows),
        adaptive_fraction=0.0,
        current_size=0,
        mstep_current_size=int(mstep_current_size),
        mstep_max_r=float(int(mstep_current_size) // 2) if mstep_max_r is None else mstep_max_r,
        image_shape=tuple(int(v) for v in image_shape),
        recon_volume_shape=tuple(int(v) for v in recon_volume_shape),
        max_adjoint_block_bytes=int(max_adjoint_block_bytes),
        stats_config=_MstepOnlyStatsConfig(n_shells=int(n_shells), n_optics_groups=int(n_optics_groups)),
        use_rfloat_ctf_wavg=recon["direct_ctf_rfloat_recon"] is not None,
        # Unshifted per-image operands take T15's kernel, which translates them
        # inside the reduction; pre-shifted tiles take the XLA statement.
        use_translate_sum_kernel=use_translate_sum_kernel,
        bpref_recon_operand=use_translate_sum_kernel and recon.get("recon_weight") is not None,
        kernel_ctf_probs=use_translate_sum_kernel and policy.kernel_ctf_probs_enabled(),
    )
    operands = resident_chunk_records.ChunkStageOperands(
        score_input=None,
        corr_img_score=None,
        highres_xi2_half=None,
        translation_prior=None,
        shifted_recon=recon.get("shifted_recon"),
        shifted_noise=recon.get("shifted_noise"),
        recon_image=recon.get("recon_image"),
        recon_weight=recon.get("recon_weight"),
        noise_image=recon.get("noise_image"),
        ctf2_over_nv_recon=recon["ctf2_over_nv_recon"],
        direct_ctf_rfloat_recon=recon["direct_ctf_rfloat_recon"],
        image_power_shells=None,
        relion_norm_high_shell=None,
        raw_translated_wavg_rectangle=recon["raw_translated_wavg_rectangle"],
        raw_translated_wavg_for_atomic=recon["raw_translated_wavg_for_atomic"],
        scale=recon["scale"],
        group_ids=None,
        translation_sqdist_ang=None,
        optics_groups=recon.get("optics_groups"),
        bpref_ctf2_over_nv_recon=recon.get("bpref_ctf2_over_nv_recon"),
    )
    tables = resident_chunk_records.ChunkStageTables(
        projection_score_cache=None,
        # P3-G: the chunk's own row arrays stand where the global pass keeps
        # its per-iteration fine-rotation caches, so the block program reads
        # its rows inside the jit instead of taking them from a host callback.
        projection_recon_cache=None if chunk_projections is None else chunk_proj,
        projection_recon_abs2_cache=(
            None if chunk_projections is None else chunk_proj_abs2
        ),
        mstep_grid=None if chunk_projections is None else chunk_mstep_rotations,
        coarse_parent_grid=None,
        fine_translation_parent=None,
        half_weights=None,
        translation_angles=(
            jnp.asarray(translation_angles, dtype=jnp.float32) if use_translate_sum_kernel else None
        ),
        full_to_compact=None,
        noise_variance_for_noise=noise_variance_for_noise,
        shell_indices_noise=shell_indices_noise,
        exact_positions=exact_positions_device,
        # The kernel addresses the reconstruction window by pixel index.
        recon_pixel_indices=(
            jnp.asarray(recon_pixel_indices, dtype=jnp.int32) if use_translate_sum_kernel else None
        ),
        relion_x_half_recon_indices=relion_x_half_recon_indices,
        shell_indices_half=None,
        wavg_shell_indices=None,
        wavg_scale_pixel_mask=None,
    )

    block_rows = int(mstep_block_rows)
    carry = None
    glue_jit = policy.resident_glue_jit_enabled()
    if chunk_projections is None:
        chunk_row_ids = None
        projection_dtypes = None
    else:
        # A host-side index vector, so it costs one transfer for the chunk and
        # no eager primitive: the block program slices it and gathers the rows.
        chunk_row_ids = (
            jnp.asarray(np.arange(int(row_capacity), dtype=np.int32))
            if row_ids is None
            else jnp.asarray(row_ids, dtype=jnp.int32)
        )
        projection_dtypes = (chunk_proj.dtype, chunk_proj_abs2.dtype)

    def start_carry(projections):
        return _initial_mstep_carry(
            Ft_y_total if row_partial is None else row_partial.y,
            Ft_ctf_total if row_partial is None else row_partial.ctf,
            operands,
            tables,
            spec=spec,
            projection_dtypes=(
                projection_dtypes
                if projections is None
                else (projections[0].dtype, projections[1].dtype)
            ),
        )

    for start in range(0, int(row_capacity), block_rows):
        if start >= int(n_valid_rows):
            # Every row of this block is chunk padding: its posterior is zero,
            # so the weighted sums, the Wavg terms, the noise partials and both
            # adjoint scatters are all exactly zero and adding them changes no
            # accumulator bit.
            break
        stop = start + block_rows
        block = slice(start, stop)
        if chunk_projections is None:
            projections = block_projections(start, stop)
        else:
            projections = None
        if carry is None:
            carry = start_carry(projections)
        if glue_jit:
            carry = _resident_mstep_block_program(
                resident_chunk_records.device_int32(start),
                resident_chunk_records.MstepBlockInputs(
                    row_image_local=row_image_local,
                    kernel_row_image_ids=kernel_row_image_ids,
                    row_posterior=row_posterior,
                    row_fine_rot=chunk_row_ids,
                    projections=projections,
                ),
                operands,
                tables,
                carry,
                spec=spec,
            )
            if row_partial is not None:
                carry, Ft_y_total, Ft_ctf_total = row_partial.block_done(
                    carry, Ft_y_total, Ft_ctf_total, block_rows=block_rows
                )
            continue
        carry = _resident_mstep_block(
            block_row_image=row_image_local[block],
            block_kernel_ids=kernel_row_image_ids[block],
            block_posterior=row_posterior[block],
            block_projections=(
                _cached_block_projections(tables, chunk_row_ids[block])
                if projections is None
                else projections
            ),
            operands=operands,
            tables=tables,
            carry=carry,
            spec=spec,
        )
        if row_partial is not None:
            carry, Ft_y_total, Ft_ctf_total = row_partial.block_done(
                carry, Ft_y_total, Ft_ctf_total, block_rows=block_rows
            )
    if carry is None:
        # A chunk with no live rows: the accumulators are the incoming ones and
        # the partials are the zeros the statistics program expects.
        carry = start_carry(
            block_projections(0, block_rows) if chunk_projections is None else None
        )
    if row_partial is not None:
        carry, Ft_y_total, Ft_ctf_total = row_partial.flush(carry, Ft_y_total, Ft_ctf_total)
    else:
        Ft_y_total, Ft_ctf_total = carry.Ft_y, carry.Ft_ctf
    carry = resident_block_programs.add_chunk_wavg_image_power(
        carry, operands, tables, row_posterior, kernel_row_image_ids, spec=spec
    )

    return (
        Ft_y_total,
        Ft_ctf_total,
        carry.wavg_triplet_pixels,
        carry.noise_shells,
        carry.a2_per_image,
        carry.xa_per_image,
    )
