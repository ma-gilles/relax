"""Device-resident K=1 sparse pass-2 driver (T9b integration).

This module wires the five stage modules of
``em_device_resident_pass2_design_20260918.md`` into one driver with the
signature and return type of
:func:`recovar.em.sparse_pass2.sparse_pass2_bucketed.compute_pass2_stats_sparse_bucketed`:

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
The driver is selected only by ``RELAX_SPARSE_PASS2_RESIDENT=1`` and only in
the production K=1 configuration (RELION x-half M-step, exact RELION fine
Gaussian scoring, float32 fine posterior, fine M-step prune, float32 scoring,
no diagnostics or dumps). Every other configuration raises
:class:`NotImplementedError` naming the missing piece; the driver never falls
back to the compact engine silently, because a silent fallback would make a
measured comparison meaningless.

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
import os
import time
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from functools import lru_cache, partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from recovar.reconstruction import noise as noise_utils

from relax.helpers.adjoint import mstep_adjoint_max_r
from relax.helpers.batch_fetch import fetch_indexed_batch
from relax.helpers.deterministic_reduce import deterministic_reductions_enabled
from relax.helpers.env_flags import parse_env_capacity_ladder, parse_env_flag
from relax.helpers.half_spectrum import (
    make_relion_noise_shell_indices_half,
    mask_relion_noise_shell_indices_to_current_window,
)
from relax.helpers.half_volume_mstep import (
    finalize_half_volume_bpref,
    half_volume_accumulator_shape,
    relion_backprojector_volume_shape,
    relion_x_half_accumulators_to_public_layout,
    relion_x_half_mstep_accumulator_dtypes,
)
from relax.helpers.optics_noise import noise_rows, pixel_rows
from relax.helpers.preprocessing import half_translation_phase_table
from relax.helpers.projection import compute_noise_block, compute_noise_block_per_optics_group
from relax.helpers.projection import (
    relion_scale_correction_pixel_mask as _relion_scale_correction_pixel_mask,
)
from relax.helpers.scale_groups import prepare_scale_correction_groups
from relax.helpers.translation_prior import (
    translation_prior_centers_for_images,
    translation_sqdist_angstrom,
    validate_translation_prior_centers,
)
from relax.helpers.types import SparsePass2Output, make_noise_stats, make_relion_stats
from relax.local.local_backprojection import (
    compute_local_ctf_sums_from_probs_sum_t,
    compute_local_weighted_sums,
)
from relax.scoring.sparse_bucket_arrays import _prepare_per_image_pass2_inputs
from relax.sparse_pass2.compile_ahead import (
    CompileAheadPool,
    resolve_compile_ahead_config,
)
from relax.sparse_pass2.resident_candidates import (
    materialize_chunk,
    merge_class_tables,
    plan_capacity_chunks,
)
from relax.sparse_pass2.resident_operands import (
    ResidentOperandsUnsupported,
    describe_resident_operand_mismatch,
    gather_resident_chunk_operands,
    prepare_resident_half_operands,
    resident_half_operand_avals,
    resident_half_operand_bytes,
    resident_half_operand_presence,
    resident_operands_max_bytes,
)
from relax.sparse_pass2.resident_scoring import score_resident_chunk
from relax.sparse_pass2.resident_significance import (
    resident_candidate_tables,
    resident_significance_csr,
)
from relax.sparse_pass2.resident_statistics import (
    FinalizedStatistics,
    ResidentStatistics,
    _drop_index,
    _flat_row_norm_and_scale_terms,
    finalize_statistics,
    make_resident_statistics,
    resident_image_capacity,
    resolve_statistics_config,
    segment_sum_by_image,
)
from relax.sparse_pass2.sparse_pass2_adjoint import _accumulate_adjoint_block_chunked
from relax.sparse_pass2.sparse_pass2_bucket_io import (
    _prepare_bucket_io,
    _relion_cuda_score_translation_angles_if_available,
)
from relax.sparse_pass2.sparse_pass2_budget import (
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
from relax.sparse_pass2.sparse_pass2_policy import (
    _RELION_WAVG_ATOMIC_SCALE_AA_ENV,
    ResidentConfigurationUnsupported,
    _fresh_k1_direct_noise_default,
    _projection_cache_enabled_for_pass,
    _relion_exact_bpref_operands_enabled,
    _relion_powerclass_spectrum_norm_enabled,
    _relion_wavg_direct_modes,
    resident_engine_selection,
)
from relax.sparse_pass2.sparse_pass2_posterior import (
    _relion_fine_parent_execution_order_enabled,
)
from relax.sparse_pass2.sparse_pass2_projection_blocks import (
    _compute_sparse_pass2_windowed_projections_block,
    _projection_kwargs_for_relion_score_window,
)
from relax.sparse_pass2.sparse_pass2_scoring import (
    _relion_cuda_fine_full_to_compact_lookup,
    _relion_native_fine_units,
    _relion_native_fine_units_enabled,
    _relion_native_score_corr_img,
    _relion_powerclass_noise_terms,
    relion_powerclass_noise_dtypes,
)
from relax.sparse_pass2.sparse_pass2_wavg import (
    _make_relion_wavg_rectangle,
    _relion_cuda_translate_wavg_norm_images,
    _relion_wavg_rectangle_image_power,
    _relion_wavg_rectangle_power_contraction,
    _relion_wavg_shifted_power,
    _replace_low_shell_noise_with_relion_wavg_direct_residual_jnp,
    image_power_shells,
    weighted_image_power_from_shells,
)
from relax.sparse_pass2.sparse_pass2_window import (
    _pass2_half_weights,
    _pass2_window_setup,
    _sparse_pass2_window_setup,
)

logger = logging.getLogger(__name__)

RESIDENT_PASS2_ENV = "RELAX_SPARSE_PASS2_RESIDENT"
_ROW_CAPACITY_LADDER_ENV = "RELAX_SPARSE_PASS2_RESIDENT_ROW_CAPACITIES"
_IMAGE_CAPACITY_LADDER_ENV = "RELAX_SPARSE_PASS2_RESIDENT_IMAGE_CAPACITIES"
_MSTEP_BLOCK_ROWS_ENV = "RELAX_SPARSE_PASS2_RESIDENT_MSTEP_BLOCK_ROWS"
# Attribution only, default off. Logs one line per chunk with its occupancy,
# its M-step block count and a device-synchronised wall, and counts the T7
# offsets readbacks. The synchronisation perturbs the wall, so an arm with
# this set is a diagnostic arm and never a timing arm.
_CHUNK_TIMING_ENV = "RELAX_SPARSE_PASS2_RESIDENT_CHUNK_TIMING"
# T14: the chunk body as one jitted program per capacity class. Opt-in; the
# per-stage path is the default and the oracle both paths are compared against. ``..._CHUNK_STATIC_BLOCKS`` runs the M-step block loop over
# the whole row capacity instead of the chunk's live blocks; both forms trace
# one program per capacity class and are bitwise equal, because a padded block
# carries a zero posterior and contributes exact zeros.
_CHUNK_JIT_ENV = "RELAX_SPARSE_PASS2_RESIDENT_CHUNK_JIT"
_CHUNK_STATIC_BLOCKS_ENV = "RELAX_SPARSE_PASS2_RESIDENT_CHUNK_STATIC_BLOCKS"
# How many M-step blocks the chunk program emits per device-loop iteration.
# XLA:GPU reads a while predicate back to the host once per iteration, so an
# unroll of u divides those readbacks by u; it also multiplies the live
# pixel-axis transients by u, because the emitted copies no longer reuse one
# block's buffers. Emitting every block (no loop) is not an option: at the early
# state's second iteration that program asked the allocator for 54.5 GiB, and
# T16's branch point 6c2dad33e carried that fully unrolled form, which OOMs at
# hp3 (181 GiB); 7a29c2776's bounded unroll supersedes it and is kept here.
_CHUNK_BLOCK_UNROLL_ENV = "RELAX_SPARSE_PASS2_RESIDENT_CHUNK_BLOCK_UNROLL"
# T16: prepare the per-image operands once per half and keep them resident, and
# take the M-step's weighted sums with T15's flat-row translate-and-sum kernel
# instead of a gathered ``[images, translations, pixels]`` tile. Default on;
# ``RELAX_SPARSE_PASS2_RESIDENT_OPERANDS=0`` selects the per-chunk
# ``_prepare_bucket_io`` preparation and the XLA tile reduction, which stay as
# the oracle both forms are compared against.
_RESIDENT_OPERANDS_ENV = "RELAX_SPARSE_PASS2_RESIDENT_OPERANDS"
# Diagnostic, default off. For the first chunk of a half it also runs the
# per-chunk preparation and checks, on that chunk's real operands, that
# translating the resident per-image arrays reproduces the pre-shifted tiles
# bitwise and that the kernel's weighted sums equal the XLA reduction. It
# doubles that chunk's preparation cost, so an arm with it set is a diagnostic
# arm, never a timing arm.
_RESIDENT_OPERANDS_VERIFY_ENV = "RELAX_SPARSE_PASS2_RESIDENT_OPERANDS_VERIFY"
# Take ``ctf_probs`` from the translate-and-sum kernel's fourth output instead
# of the XLA statement. Measurement only: see
# ``_resident_block_weighted_sums_kernel`` for why it is not the default.
_KERNEL_CTF_PROBS_ENV = "RELAX_SPARSE_PASS2_RESIDENT_KERNEL_CTF_PROBS"
# P4-G phase 2: square the chunk's Wavg rectangle once per image and gather the
# float32 result, instead of gathering the complex rectangle to the block's rows
# and squaring once per row. Squaring is elementwise, so squaring then gathering
# and gathering then squaring are the same float32 values, and the contraction
# that follows keeps its shapes and its reduction order; the two settings are
# bitwise. Default off while the measurement arms are the ones in the ticket's
# report; `_resident_block_wavg_rectangle_terms` holds both paths.
_WAVG_POWER_PER_IMAGE_ENV = "RELAX_SPARSE_PASS2_RESIDENT_WAVG_POWER_PER_IMAGE"
# P3-A: dispatch the per-stage chunk loop's three stages as jitted programs
# keyed on the capacity class instead of as loose eager operations. Default on.
# ``RELAX_SPARSE_PASS2_RESIDENT_GLUE_JIT=0`` restores the loose dispatch,
# which stays the oracle every bitwise comparison of this change is made
# against. The stage bodies are the same functions in both settings, so the
# flag changes only where the JIT boundary sits.
_RESIDENT_GLUE_JIT_ENV = "RELAX_SPARSE_PASS2_RESIDENT_GLUE_JIT"
# Diagnostic, default off. Checks the statically computed M-step carry avals
# against a ``jax.eval_shape`` probe of the same block stages, once per
# capacity class. The probes are what this change removes from the chunk loop;
# the flag exists so a test, or a suspicious run, can prove the arithmetic
# still agrees with them.
_CARRY_AVAL_PROBE_ENV = "RELAX_SPARSE_PASS2_RESIDENT_CARRY_AVAL_PROBE"
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
# per capacity-class stage, and with the chunk program one more program again.
# The occupancy of each plan is still logged, so a future change has its number.
_DEFAULT_ROW_CAPACITY_LADDER = (8192, 32768, 131072)
_DEFAULT_IMAGE_CAPACITY_LADDER = (32, 128, 512)

__all__ = [
    "RESIDENT_PASS2_ENV",
    "ResidentClassInputs",
    "ResidentKClassPass2Output",
    "ResidentPass2Plan",
    "compute_k_class_pass2_stats_resident",
    "compute_pass2_stats_resident",
    "resident_pass2_requested",
    "require_resident_production_configuration",
    "resident_pass2_out_of_scope_reason",
]


def resident_pass2_requested() -> bool:
    """Return whether the device-resident K=1 pass 2 is selected (the default).

    ``RELAX_SPARSE_PASS2_RESIDENT=0`` selects the compact engine for A/B checks;
    configurations outside the resident scope route to compact either way
    (:func:`resident_pass2_out_of_scope_reason`).
    """

    return resident_engine_selection(RESIDENT_PASS2_ENV) != "off"


# ---------------------------------------------------------------------------
# Production-configuration gate
# ---------------------------------------------------------------------------

_DIAGNOSTIC_DIR_ENVS = (
    "RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR",
    "RELAX_BPREF_CONTRIBUTION_DUMP_DIR",
    "RELAX_BPREF_MEMBERSHIP_DUMP_DIR",
    "RELAX_PASS2_DUMP_DIR",
    "RELAX_BPREF_EXECUTION_ORDER_LOCAL_FILE",
    "RELAX_VDAM_KCLASS_STATS_DUMP_DIR",
)

_DIAGNOSTIC_FLAG_ENVS = (
    "RELAX_PASS2_DUMP_NORM_RESIDUAL_INPUTS",
    "RELAX_K1_RELION_TRANSLATED_WAVG_NORM",
    "RELAX_SPARSE_PASS2_LOG_CANDIDATE_DENSITY",
    "RELAX_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION",
    "RELAX_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH",
    "RELAX_RELION_X_HALF_BP_FUSED_ATOMICS",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ResidentConfigurationUnsupported(
            "The device-resident K=1 sparse pass 2 "
            f"({RESIDENT_PASS2_ENV}=1) does not implement this configuration: {message}. "
            "Clear the flag to use the compact engine; this path never falls back silently."
        )


def _resident_wavg_arithmetic(
    *,
    accumulate_noise,
    scale_groups_available,
    preserve_bpref_particle_order,
    source_faithful_spectrum_norm,
):
    """Resolve the fresh-K=1 Wavg arithmetic exactly as the compact engine does.

    Returns ``(spectrum_norm, exact_bpref_operands, direct_noise_default,
    atomic_scale_aa)``. The resident statistics stage implements only the
    atomic Wavg triplet, which the compact engine selects inside the fresh
    K=1 guard (``source_faithful_spectrum_norm``, i.e. a run that does not
    replay RELION's BPref particle order).
    """

    fresh_k1_guard = bool(source_faithful_spectrum_norm)
    spectrum_norm = _relion_powerclass_spectrum_norm_enabled(fresh_k1_guard=fresh_k1_guard)
    exact_bpref_operands = _relion_exact_bpref_operands_enabled(
        fresh_k1_guard=fresh_k1_guard,
        source_faithful_spectrum_norm=spectrum_norm,
    )
    direct_noise_default = _fresh_k1_direct_noise_default(
        preserve_bpref_particle_order=preserve_bpref_particle_order,
        relion_exact_bpref_operands=exact_bpref_operands,
    )
    atomic_scale_aa = bool(
        accumulate_noise
        and scale_groups_available
        and parse_env_flag(_RELION_WAVG_ATOMIC_SCALE_AA_ENV, default=direct_noise_default)
    )
    return spectrum_norm, exact_bpref_operands, direct_noise_default, atomic_scale_aa


def resident_pass2_out_of_scope_reason(
    *,
    accumulate_noise=False,
    scale_groups_available=False,
    preserve_bpref_particle_order=False,
    source_faithful_spectrum_norm=False,
) -> str | None:
    """Name the pass-2 routes the resident driver was never scoped to cover.

    These are not configuration drift inside the covered path, so they are not
    a reason to stop a run. The caller sends them to the compact engine and
    says so:

    - a production-shaped pass (noise and group-scale statistics) that does not
      preserve RELION's particle order (a subset or focused debugging replay)
      uses the compact engine's unordered, non-atomic Wavg arithmetic; the
      resident statistics stage implements only the atomic triplet.

    A fresh K=1 pass at any box is in scope: the resident driver scores its
    fine diff2 in RELION's native FFT units, as the compact engine does.

    Everything else still raises through
    :func:`require_resident_production_configuration`, because a silent
    fallback there would hide a real mismatch.
    """

    if accumulate_noise and scale_groups_available:
        atomic_scale_aa = _resident_wavg_arithmetic(
            accumulate_noise=accumulate_noise,
            scale_groups_available=scale_groups_available,
            preserve_bpref_particle_order=preserve_bpref_particle_order,
            source_faithful_spectrum_norm=source_faithful_spectrum_norm,
        )[3]
        if not atomic_scale_aa:
            return (
                "the non-atomic Wavg arithmetic of a pass without RELION's "
                "preserved particle order (subset or focused replay)"
            )
    return None


def require_resident_production_configuration(**kwargs) -> None:
    """Raise :class:`NotImplementedError` unless this is the production path.

    Every check names the specific missing piece rather than reporting a
    generic refusal, so a caller that trips one knows which behaviour the
    resident driver would have to grow.
    """

    _require(bool(kwargs["relion_x_half_mstep"]), "the RELION x-half M-step is required")
    # The dispatcher opens a persistent texture only for the compact engine; a
    # direct caller that supplies one gets a named refusal, not a drop.
    _require(
        kwargs["relion_projector_texture"] is None,
        "a persistent RELION projector texture belongs to the compact engine; "
        "the resident driver projects from relion_projector_half",
    )
    score_mode = kwargs["relion_firstiter_score_mode"]
    _require(
        (score_mode == "gaussian" and bool(kwargs["relion_exact_fine_gaussian"]))
        or (score_mode == "normalized_cc" and bool(kwargs.get("relion_exact_fine_normalized_cc"))),
        "exact RELION fine Gaussian scoring, or RELION's literal fine normalized-CC "
        "reduction for --firstiter_cc, is required "
        f"(got relion_exact_fine_gaussian={kwargs['relion_exact_fine_gaussian']!r}, "
        f"relion_exact_fine_normalized_cc={kwargs.get('relion_exact_fine_normalized_cc')!r}, "
        f"score_mode={score_mode!r})",
    )
    _require(not bool(kwargs["use_float64_scoring"]), "float64 scoring is a diagnostic mode")
    # RELION's --firstiter_cc iteration scores with normalized CC and keeps only
    # the best weight (ml_optimiser.cpp:9266-9293); a Gaussian pass never does.
    _require(
        bool(kwargs["relion_firstiter_winner_take_all"]) == (score_mode == "normalized_cc"),
        "winner-take-all goes with the --firstiter_cc normalized-CC pass and only with it "
        f"(winner_take_all={bool(kwargs['relion_firstiter_winner_take_all'])!r}, "
        f"score_mode={score_mode!r})",
    )
    _require(
        not (bool(kwargs["disable_adjoint_y"]) or bool(kwargs["disable_adjoint_ctf"])),
        "score-only passes have no M-step to make resident",
    )
    _require(not bool(kwargs["return_score_log_z_only"]), "score-logZ-only passes are score-only")
    _require(bool(kwargs["accumulate_noise"]), "the production pass accumulates noise statistics")
    # The K=1 adaptive route always hands the M-step call an
    # ``normalization_other_score_log_z`` built from the *other* classes'
    # log-Z (k_class.py::_run_sparse_k_class_adaptive_pass2). At K=1 there are
    # no other classes, so that vector is all -inf and the compact engine's
    # own arithmetic collapses to its unnormalized branch: logaddexp(x, -inf)
    # is x, the reported log-evidence and score log-Z are taken from
    # ``local_score_log_z`` rather than from the combined value, and the
    # float32 reconstruction weights never read it at all. Accept exactly that
    # degenerate vector, which is the production K=1 case, and refuse any
    # finite entry, which would genuinely mix classes.
    other_log_z = kwargs["normalization_other_score_log_z"]
    other_log_z_is_degenerate = other_log_z is not None and bool(
        np.all(np.asarray(other_log_z) == -np.inf)
    )
    _require(
        kwargs["normalization_log_z"] is None,
        "an externally supplied log-Z belongs to the K-class engine",
    )
    _require(
        other_log_z is None or other_log_z_is_degenerate,
        "a finite cross-class score normalization belongs to the K-class engine",
    )
    coarse_sum_weight = kwargs["relion_f32_normalization_sum_weight"]
    coarse_winner = kwargs["relion_coarse_hard_assignment"]
    coarse_max_posterior = kwargs.get("relion_coarse_max_posterior")
    _require(
        (coarse_sum_weight is None) == (coarse_winner is None) == (coarse_max_posterior is None),
        "the zero-oversampling pass reuses the coarse normalization sum, winner and Pmax "
        "together, as the K=1 adaptive route supplies them",
    )
    _require(
        coarse_sum_weight is None or int(kwargs.get("oversampling_order", 0)) == 0,
        "the coarse normalization sum is reused only at zero oversampling "
        "(acc_ml_optimiser_impl.h:2868)",
    )
    # ``preserve_bpref_particle_order`` is the production setting, and on its
    # own it forces one BPref launch per particle. The production run pairs it
    # with the soft-posterior block prototype, which turns those launches back
    # into one block launch per bucket; that is the semantics the resident
    # driver reproduces with one launch per row block. Without the prototype
    # the compact engine really would launch per particle, so refuse.
    _require(
        (not bool(kwargs["preserve_bpref_particle_order"]))
        or bool(kwargs["soft_posterior_block_bpref"]),
        "strict per-particle BPref launches are not implemented; set "
        "RELAX_EM_PROTOTYPE_SOFT_POSTERIOR_BLOCK_BPREF=1 (the production "
        "setting) so BPref accumulates per block, or clear "
        "preserve_bpref_particle_order",
    )
    _require(
        kwargs["fine_rotations_override"] is not None
        and kwargs["fine_rotation_parent_override"] is not None,
        "the resident driver gathers rotations from the caller's fine grid, "
        "so fine_rotations_override and fine_rotation_parent_override are required",
    )
    _require(
        bool(kwargs["use_window"]),
        "the resident driver scores through the RELION current-size window; "
        "a full-half pass would include the ky=-N/2 Nyquist row, where the "
        "fused-translate and pre-shifted scorers are known to differ",
    )
    _require(
        bool(kwargs["projection_cache_available"]),
        "the resident scoring stage gathers cached fine-rotation projections; "
        "the per-iteration projection cache is disabled or did not fit its budget",
    )
    _require(
        bool(kwargs["relion_wavg_atomic_scale_aa"]),
        "the resident statistics stage consumes the RELION atomic Wavg triplet",
    )
    _require(
        bool(kwargs["relion_wavg_atomic_direct_noise"]),
        "the resident statistics stage uses RELION's direct low-shell residual",
    )
    _require(
        not bool(kwargs["relion_wavg_atomic_direct_norm"]),
        "the direct per-particle Wavg norm arm is a stopped diagnostic",
    )
    for name in _DIAGNOSTIC_DIR_ENVS:
        _require(
            not os.environ.get(name, "").strip(),
            f"the diagnostic dump {name} is set; the resident driver emits no dumps",
        )
    for name in _DIAGNOSTIC_FLAG_ENVS:
        _require(
            not parse_env_flag(name, default=False),
            f"the diagnostic flag {name} is set; the resident driver has no such arm",
        )


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidentPass2Plan:
    """Chunk plan plus the pixel-axis block size the M-step stages run at."""

    chunks: tuple
    row_capacity_ladder: tuple
    image_capacity_ladder: tuple
    mstep_block_rows: int


def _floor_power_of_two(value: int) -> int:
    value = int(value)
    if value <= 1:
        return 1
    return 1 << (value.bit_length() - 1)


def _resolve_mstep_block_rows(
    *,
    n_recon_pixels: int,
    max_block_bytes: int,
    row_capacity_ladder: tuple,
) -> int:
    """Rows per pixel-axis block: a power of two dividing every row capacity.

    A block holds, per row and reconstruction pixel, one complex64 projection,
    one float32 ``|proj|^2``, two complex64 weighted sums, one float32 CTF sum
    and the float32 Wavg triplet: 44 bytes. Sizing from the same budget the
    compact engine uses for its adjoint blocks keeps the transient comparable
    to the bucket this replaces; quantizing to a power of two keeps the traced
    pixel-axis program count at one per pixel count.
    """

    override = os.environ.get(_MSTEP_BLOCK_ROWS_ENV, "").strip()
    if override:
        block = int(override)
        if block <= 0 or block & (block - 1):
            raise ValueError(f"{_MSTEP_BLOCK_ROWS_ENV} must be a positive power of two, got {block}")
    else:
        bytes_per_row = max(int(n_recon_pixels), 1) * 44
        block = _floor_power_of_two(max(int(max_block_bytes) // bytes_per_row, 1))
    smallest = int(row_capacity_ladder[0])
    block = min(block, smallest)
    while block > 1 and smallest % block:
        block //= 2
    return max(block, 1)


def _cap_image_capacity_ladder(
    ladder: tuple,
    *,
    n_fine_trans: int,
    n_recon_pixels: int,
    max_tile_bytes: int,
) -> tuple:
    """Drop image classes whose translation tiles exceed the tile budget.

    A chunk materializes three ``(images, T, P)`` complex64 tiles: the
    reconstruction operand, the noise operand and RELION's Wavg rectangle.
    """

    per_image = max(int(n_fine_trans), 1) * max(int(n_recon_pixels), 1) * 8 * 3
    cap = max(int(max_tile_bytes) // max(per_image, 1), 1)
    kept = tuple(value for value in ladder if int(value) <= cap)
    return kept if kept else (int(ladder[0]),)


# ---------------------------------------------------------------------------
# Device programs with a pixel axis (one per (block rows, pixel count) pair)
# ---------------------------------------------------------------------------


@jax.jit
def _resident_block_weighted_sums(
    row_posterior,  # float32 [block, T]
    row_image_local,  # int32 [block]
    shifted_recon,  # complex [C_B, T, P]
    shifted_noise,  # complex [C_B, T, P]
    ctf2_over_nv_recon,  # real [C_B, P]
):
    """Flat-row twin of the bucket's ``compute_local_mstep_sums`` pair.

    ``compute_local_weighted_sums`` contracts ``(B, R, T) x (B, T, N)`` with
    ``Precision.HIGHEST``; the flat-row form gathers each row's image tile and
    contracts ``(Q, T) x (Q, T, N)`` at the same precision, so the per-cell
    products and the translation reduction order are unchanged.
    ``compute_local_ctf_sums_from_probs_sum_t`` is reused verbatim, including
    its ``!= 0`` mass predicate. ``summed`` feeds the x-half BPref numerator;
    ``summed_masked`` is the noise operand the host tail builds from
    ``shifted_noise``.
    """

    row_image_local = jnp.asarray(row_image_local, dtype=jnp.int32)
    weights = jnp.asarray(row_posterior)
    recon_tiles = jnp.asarray(shifted_recon)[row_image_local]
    noise_tiles = jnp.asarray(shifted_noise)[row_image_local]
    # ``compute_local_weighted_sums`` is called verbatim with a singleton
    # rotation axis, so the contraction keeps its pinned
    # ``Precision.HIGHEST``; only the gathered per-row tile replaces the
    # shared per-image tile.
    summed = compute_local_weighted_sums(weights[:, None, :], recon_tiles)[:, 0, :]
    summed_masked = compute_local_weighted_sums(weights[:, None, :], noise_tiles)[:, 0, :]
    probs_sum_t = jnp.sum(weights, axis=-1)
    # The rectangular helper takes ``(B, R)`` rotation sums against one
    # ``(B, P)`` CTF row per image. In the flat-row layout each row carries its
    # own gathered CTF row, so the call is made with a singleton rotation axis;
    # the ``!= 0`` mass predicate and the product order are unchanged.
    ctf_probs = compute_local_ctf_sums_from_probs_sum_t(
        probs_sum_t[:, None],
        jnp.asarray(ctf2_over_nv_recon)[row_image_local],
    )[:, 0, :]
    return summed, summed_masked, ctf_probs, probs_sum_t


def _resident_block_weighted_sums_kernel(
    row_posterior,  # float32 [block, T]
    row_image_ids,  # int32 [block], -1 on a padded row
    row_image_local,  # int32 [block], the chunk-local slot the XLA gather uses
    recon_image,  # complex64 [C_B, P], unshifted
    recon_weight,  # float32 [C_B, P] (BPref weighted CTF) or None
    noise_image,  # complex64 [C_B, P], unshifted
    ctf2_over_nv_recon,  # float32 [C_B, P]
    recon_pixel_indices,  # int32 [P], centered packed-half indices
    translation_angles,  # float32 [T, 2]
    *,
    image_shape,
    n_recon_pixels: int,
    kernel_ctf_probs: bool,
    cuda_backproject,
):
    """T15's translate-and-sum kernel in place of the gathered-tile reduction.

    Same four outputs as :func:`_resident_block_weighted_sums`, from the
    *unshifted* per-image operands: the kernel applies each translation inside
    the reduction with the phase arithmetic of the primitive that built the
    tile, so no ``[images, translations, pixels]`` tile exists. ``recon_weight``
    selects the convention pairing production uses -- BPref for the
    reconstruction operand, score for the noise operand -- and matches how
    ``_prepare_bucket_io`` builds the two shifted arrays.

    Rows are bounded by their image id rather than by a row count: a padded row
    carries ``-1`` and the kernel writes it as zeros, which is the value the XLA
    path reaches through a zero posterior.

    ``ctf_probs``. The kernel can also produce it, from its own sequential
    ``probs_sum_t``. That mass is bitwise against ``jnp.sum`` only where XLA
    happens to reduce the translations sequentially too: it does at
    ``[2048, 21]``, and it does not at ``[64, 21]``, where the fourth output
    moves 53% of the cells by up to 1 relative ulp. ``ctf_probs`` feeds the
    ``Ft_ctf`` accumulator and the Wavg and noise terms, so the default keeps
    the XLA statement the per-chunk path used, on the same gathered CTF row;
    the fused form stays selectable for measurement, and the kernel's own mass
    is never used for anything else.
    """

    row_posterior = jnp.asarray(row_posterior, dtype=jnp.float32)
    block_rows = int(row_posterior.shape[0])
    outputs = cuda_backproject.relion_translate_sum_flat_rows_f32(
        jnp.asarray(recon_image, dtype=jnp.complex64),
        jnp.asarray(noise_image, dtype=jnp.complex64),
        jnp.asarray(row_image_ids, dtype=jnp.int32),
        row_posterior,
        jnp.asarray(translation_angles, dtype=jnp.float32),
        jnp.asarray(recon_pixel_indices, dtype=jnp.int32),
        jnp.asarray(block_rows, dtype=jnp.int32),
        jnp.asarray(int(n_recon_pixels), dtype=jnp.int32),
        recon_weight=(
            None if recon_weight is None else jnp.asarray(recon_weight, dtype=jnp.float32)
        ),
        ctf2_over_nv=(
            jnp.asarray(ctf2_over_nv_recon, dtype=jnp.float32) if kernel_ctf_probs else None
        ),
        image_shape=tuple(int(size) for size in image_shape),
    )
    if kernel_ctf_probs:
        summed, summed_masked, probs_sum_t, ctf_probs = outputs
        return summed, summed_masked, ctf_probs, probs_sum_t
    summed, summed_masked, _kernel_mass = outputs
    ctf_probs, probs_sum_t = _resident_block_ctf_probs(
        row_posterior, row_image_local, ctf2_over_nv_recon
    )
    return summed, summed_masked, ctf_probs, probs_sum_t


@jax.jit
def _resident_block_ctf_probs(row_posterior, row_image_local, ctf2_over_nv_recon):
    """``ctf_probs`` and its mass, exactly as :func:`_resident_block_weighted_sums` forms them.

    One program, so the kernel path costs one dispatch here rather than three.
    The statements, the gather and the ``!= 0`` mass predicate are the tile
    path's own, which is what makes the two paths bitwise on this output.
    """

    probs_sum_t = jnp.sum(jnp.asarray(row_posterior), axis=-1)
    ctf_probs = compute_local_ctf_sums_from_probs_sum_t(
        probs_sum_t[:, None],
        jnp.asarray(ctf2_over_nv_recon)[jnp.asarray(row_image_local, dtype=jnp.int32)],
    )[:, 0, :]
    return ctf_probs, probs_sum_t


@partial(jax.jit, static_argnums=1, donate_argnums=0)
def _relion_native_fine_units_in_place(values, fft_size):
    """:func:`_relion_native_fine_units` as one program that reuses the input buffer.

    The whole-grid projection cache is several GiB; the eager form holds the
    complex64 input, two float64 part arrays and the output at once (EMPIAR-10097
    VDAM at healpix 3, current size 56: a 7.77 GiB float64 temporary ran the
    device out of memory, job 14422223). The fused program has no float64
    temporaries and writes into the donated input. Same elementwise statements,
    so the same values.
    """

    return _relion_native_fine_units(values, fft_size)


@jax.jit
def _resident_block_residual(summed, probs_sum_t, proj, ctf2_over_nv_recon, row_image_local):
    """VDAM's BPref numerator: the weighted image sum minus the weighted CTF'd projection.

    RELION's ``--grad`` backprojection (``cuda_kernel_backproject3D_SGD``,
    acc/cuda/cuda_kernels/BP.cuh:406-560) accumulates ``(shift(img) - ctf * proj) * w``
    per translation. The projection does not depend on the translation, so the sum
    is ``summed - (sum_t w) * ctf^2/sigma2 * proj``: the same statement as the exact
    local engine's residual (``local_big_jit``), with its ``!= 0`` mass predicate.
    """

    frefctf_weighted = jnp.asarray(proj) * jnp.asarray(ctf2_over_nv_recon)[
        jnp.asarray(row_image_local, dtype=jnp.int32)
    ]
    probs_sum_t = jnp.asarray(probs_sum_t)
    frefctf_delta = jnp.where(
        probs_sum_t[:, None] != 0.0,
        probs_sum_t[:, None] * frefctf_weighted,
        0.0,
    )
    return summed - frefctf_delta


@partial(jax.jit, static_argnames=("n_shells", "image_capacity"))
def _resident_block_noise_and_norm(
    proj,  # complex [block, P]
    proj_abs2,  # real [block, P]
    summed_masked,  # complex [block, P]
    ctf_probs,  # real [block, P]
    noise_variance,  # real [P], or [G, P] per optics group
    shell_indices,  # int32 [P]
    row_image_local,  # int32 [block]
    row_optics_groups=None,  # int32 [block] with a [G, P] noise table
    *,
    n_shells: int,
    image_capacity: int,
):
    """One row block's noise shells plus its per-image ``A2``/``XA`` partials.

    With a per-optics-group noise table each row uses its image's group spectrum
    and the shells come back per group, ``[G, n_shells]``.
    """

    if row_optics_groups is None:
        block_noise_shells, _, _ = compute_noise_block(
            proj,
            proj_abs2,
            summed_masked,
            ctf_probs,
            noise_variance,
            shell_indices,
            int(n_shells),
            return_split=False,
        )
        row_noise = jnp.asarray(noise_variance)
    else:
        block_noise_shells = compute_noise_block_per_optics_group(
            proj,
            proj_abs2,
            summed_masked,
            ctf_probs,
            noise_variance,
            row_optics_groups,
            shell_indices,
            int(n_shells),
        )
        row_noise = jnp.asarray(noise_variance)[row_optics_groups]
    a2_per_row, xa_per_row = _flat_row_norm_and_scale_terms(
        proj, proj_abs2, summed_masked, ctf_probs, row_noise
    )
    a2_per_image = segment_sum_by_image(a2_per_row, row_image_local, int(image_capacity))
    xa_per_image = segment_sum_by_image(xa_per_row, row_image_local, int(image_capacity))
    return block_noise_shells.astype(jnp.float64), a2_per_image, xa_per_image


@jax.jit
def _resident_block_wavg_algebraic_terms(
    proj,  # complex [block, P]
    proj_abs2,  # real [block, P]
    summed_masked,  # complex [block, P]
    ctf_probs,  # real [block, P]
    noise_variance,  # real [P]
    scale,  # real [C_B]
    raw_shifted_images,  # complex64 [C_B, T, P]
    row_posterior,  # float32 [block, T]
    row_image_local,  # int32 [block]
    row_optics_groups=None,  # int32 [block] with a [G, P] noise table
):
    """Flat-row twin of ``_relion_wavg_atomic_triplet_terms``.

    Used when the pass does not carry RELION's RFLOAT CTF operand, which is
    the branch the host bucket tail takes when ``direct_ctf_rfloat_recon`` is
    ``None``. Term for term it is the rectangular helper with the
    ``(image, rotation)`` axes folded into the row axis: the two ``!= 0`` mass
    predicates, the per-image scale division, the float32 casts and the
    optimization barrier between the real and imaginary image-power halves all
    stay where they are.
    """

    proj = jnp.asarray(proj, dtype=jnp.complex64)
    proj_abs2 = jnp.asarray(proj_abs2, dtype=jnp.float32)
    summed_masked = jnp.asarray(summed_masked, dtype=jnp.complex64)
    ctf_probs = jnp.asarray(ctf_probs, dtype=jnp.float32)
    noise_variance = (
        jnp.asarray(noise_variance, dtype=jnp.float32).reshape(-1)[None, :]
        if row_optics_groups is None
        else jnp.asarray(noise_variance, dtype=jnp.float32)[row_optics_groups]
    )
    row_image_local = jnp.asarray(row_image_local, dtype=jnp.int32)
    row_scale = jnp.asarray(scale, dtype=jnp.float32).reshape(-1)[row_image_local]
    posterior = jnp.asarray(row_posterior, dtype=jnp.float32)

    ctf_has_mass = ctf_probs != 0.0
    ctf_posterior_raw = jnp.where(ctf_has_mass, ctf_probs * noise_variance, 0.0)
    aa_raw = jnp.where(ctf_has_mass, proj_abs2 * ctf_posterior_raw, 0.0).astype(jnp.float32)
    cross_has_mass = summed_masked != 0.0
    cross = jnp.where(cross_has_mass, proj * jnp.conj(summed_masked), 0.0)
    xa_raw = (noise_variance * cross.real).astype(jnp.float32)
    safe_scale = jnp.maximum(row_scale, jnp.asarray(1e-30, dtype=jnp.float32))
    xa = (xa_raw / safe_scale[:, None]).astype(jnp.float32)
    aa = (aa_raw / (safe_scale[:, None] ** 2)).astype(jnp.float32)

    tiles = jnp.asarray(raw_shifted_images, dtype=jnp.complex64)[row_image_local]
    image_power = _relion_wavg_rectangle_image_power(tiles, posterior[:, None, :])[:, 0, :]
    diff2 = (
        (image_power + aa_raw) - jnp.asarray(2.0, dtype=jnp.float32) * xa_raw
    ).astype(jnp.float32)
    return jnp.stack((xa, aa, diff2), axis=-1)


@partial(jax.jit, static_argnames=("power_per_image",))
def _resident_block_wavg_rectangle_terms(
    exact_terms,  # float32 [block, P_exact, 3]
    raw_shifted_rectangle,  # complex64 [C_B, T, P_rect]
    row_posterior,  # float32 [block, T]
    row_image_local,  # int32 [block]
    exact_positions,  # int32 [P_exact]
    *,
    power_per_image: bool = False,
):
    """Flat-row twin of ``_relion_wavg_rectangle_triplet_terms``.

    Same two statements as the rectangular helper: fill the whole rectangle's
    ``diff2`` slot with RELION's posterior-weighted image power, then overwrite
    the exact-radius positions with the projected triplet.

    ``power_per_image`` moves the squaring to the other side of the gather.
    ``|x|^2`` depends only on the image's rectangle, so with the flag on it is
    computed once for the chunk's ``C_B`` images and the float32 result is
    gathered to the block's rows; with it off the complex rectangle is gathered
    first and squared once per row. Squaring is elementwise, so both orders give
    the same float32 values, and the contraction that consumes them keeps the
    same operand shapes and the same reduction over the translation axis: the
    two settings are bitwise. The block is ``mstep_block_rows`` rows (2048 in
    production) over an image capacity of 32 or 128, so the flag removes 16x to
    64x of the squaring and halves the gathered bytes, float32 rather than
    complex64. P4-G measured that squaring at 2.011 s of a 4.5 s hp3 replay
    iteration.
    """

    row_image_local = jnp.asarray(row_image_local, dtype=jnp.int32)
    block_posterior = jnp.asarray(row_posterior, dtype=jnp.float32)[:, None, :]
    if power_per_image:
        image_power = _relion_wavg_rectangle_power_contraction(
            _relion_wavg_shifted_power(raw_shifted_rectangle)[row_image_local],
            block_posterior,
        )[:, 0, :]
    else:
        tiles = jnp.asarray(raw_shifted_rectangle, dtype=jnp.complex64)[row_image_local]
        image_power = _relion_wavg_rectangle_image_power(tiles, block_posterior)[:, 0, :]
    rectangle_terms = jnp.zeros(image_power.shape + (3,), dtype=jnp.float32)
    rectangle_terms = rectangle_terms.at[..., 2].set(image_power)
    return rectangle_terms.at[:, jnp.asarray(exact_positions, dtype=jnp.int32), :].set(
        jnp.asarray(exact_terms, dtype=jnp.float32)
    )


# ---------------------------------------------------------------------------
# Per-chunk image-level statistics (every term without a row-pixel axis)
# ---------------------------------------------------------------------------


class _ChunkImageOperands(NamedTuple):
    """One chunk's image-level statistics operands, all already on the device."""

    row_posterior: jax.Array  # float32 [C_R, T]
    row_image_local: jax.Array  # int32 [C_R]
    row_coarse_rot: jax.Array  # int32 [C_R], padded -> >= n_coarse_rot
    image_ids: jax.Array  # int32 [C_B], padded -> -1
    group_ids: jax.Array  # int32 [C_B], padded -> -1
    image_power_shells: jax.Array  # float64 [C_B, n_shells], each image's power per noise shell
    relion_norm_high_shell: jax.Array  # real [C_B]
    wavg_triplet_pixels: jax.Array  # float32 [C_B, P_rect, 3]
    block_noise_shells: jax.Array  # float64 [n_shells]
    a2_per_image: jax.Array  # real [C_B]
    xa_per_image: jax.Array  # real [C_B]
    class_log_z: jax.Array  # float64 [C_B]
    min_diff2: jax.Array  # real [C_B]
    best_log_score: jax.Array  # float32 [C_B]
    max_posterior: jax.Array  # float32 [C_B]
    best_cell_index: jax.Array  # int64 [C_B], segment-relative (r_local * T + t)
    best_fine_rot: jax.Array  # int64 [C_B], global fine rotation id of the winner
    optics_groups: jax.Array | None = None  # int32 [C_B] with G > 1 optics groups
    # K>1 only: each row's class and the (slot, class) reductions, flat slot * K + class.
    row_class: jax.Array | None = None  # int32 [C_R]
    per_class_log_z: jax.Array | None = None  # float64 [C_B * K]
    per_class_best_log_score: jax.Array | None = None  # float32 [C_B * K]
    per_class_best_cell: jax.Array | None = None  # int64 [C_B * K], fine rotation * T + t, -1 if none
    # K>1: each image's scale sums under its classes' own masks (_fold_class_scale_sums).
    scale_xa_per_image: jax.Array | None = None  # float64 [C_B]
    scale_aa_per_image: jax.Array | None = None  # float64 [C_B]


class _ChunkImageTables(NamedTuple):
    """Iteration-global tables the image-level program reads every chunk."""

    shell_indices_half: jax.Array  # int32 [P_half]
    wavg_shell_indices: jax.Array  # int32 [P_rect]
    wavg_scale_pixel_mask: jax.Array  # bool [P_rect]; [K, P_rect] with K>1 classes
    translation_sqdist_ang: jax.Array | None  # real [T] or [C_B, T] or None


@partial(jax.jit, static_argnames=("config",))
def _accumulate_chunk_image_terms(
    stats: ResidentStatistics,
    operands: _ChunkImageOperands,
    tables: _ChunkImageTables,
    *,
    config,
) -> ResidentStatistics:
    """Fold one chunk's image-level terms into the device accumulators.

    Statement for statement this is the host bucket tail in its production
    configuration; the only change is that the row-pixel reductions arrive as
    already-summed chunk partials, because the pixel axis is walked in blocks.
    """

    n_shells = int(config.n_shells)
    n_fine_trans = int(config.n_fine_trans)
    image_capacity = int(operands.image_ids.shape[0])

    probs = operands.row_posterior
    row_image = jnp.asarray(operands.row_image_local, dtype=jnp.int32)
    image_ids = jnp.asarray(operands.image_ids, dtype=jnp.int32)
    valid_image = image_ids >= 0
    image_slot = _drop_index(image_ids, int(config.image_capacity))

    # --- 1/2. sigma2 offset and support mass -------------------------------
    translation_posterior = segment_sum_by_image(probs, row_image, image_capacity)
    sigma2_offset = stats.sigma2_offset
    if tables.translation_sqdist_ang is not None:
        sqdist = jnp.asarray(tables.translation_sqdist_ang, dtype=jnp.float64)
        sigma2_offset = sigma2_offset + jnp.sum(
            translation_posterior.astype(jnp.float64) * sqdist
        )
    support_mass = jnp.sum(translation_posterior, axis=1)
    support_mass = jnp.where(valid_image, support_mass, jnp.zeros((), support_mass.dtype))
    n_optics_groups = int(config.n_optics_groups)
    if n_optics_groups == 1:
        sumw = stats.sumw + jnp.sum(support_mass.astype(jnp.float64))
    else:
        # RELION's sumw_group[optics_group]: the support mass of each group's images.
        image_optics = jnp.asarray(operands.optics_groups, dtype=jnp.int32)
        sumw = stats.sumw + jax.ops.segment_sum(
            support_mass.astype(jnp.float64), image_optics, num_segments=n_optics_groups
        )

    # --- 3. weighted image power shells and per-image norm power -----------
    weighted_img_shells, weighted_img_per_image = weighted_image_power_from_shells(
        operands.image_power_shells,
        support_mass,
        operands.relion_norm_high_shell,
        valid_image,
        norm_unweighted_shell_cutoff=config.norm_unweighted_shell_cutoff,
        include_unweighted_high_shell=config.include_unweighted_high_shell,
        deterministic_norm_reduction=config.deterministic_norm_reduction,
    )

    # --- 4. per-particle norm correction (production algebraic mode) -------
    # The host adds the image-power term and the A2-2XA residual in two
    # separate ``+=`` statements; keep both scatters separate.
    norm_correction = stats.norm_correction.at[image_slot].add(
        jnp.where(valid_image, weighted_img_per_image, 0.0).astype(jnp.float64),
        mode="drop",
    )

    # --- 5/6. noise shells with RELION's direct low-shell replacement ------
    if n_optics_groups == 1:
        residual_shells, img_power_shells = (
            _replace_low_shell_noise_with_relion_wavg_direct_residual_jnp(
                jnp.asarray(operands.block_noise_shells, dtype=jnp.float64),
                weighted_img_shells.astype(jnp.float64),
                operands.wavg_triplet_pixels[:, :, 2],
                tables.wavg_shell_indices,
                exclusive_shell_stop=int(config.direct_noise_exclusive_shell_stop),
                shell_count=n_shells,
            )
        )
    else:
        # The same two steps once per optics group, over that group's images only.
        residual_per_group, power_per_group = [], []
        for group in range(n_optics_groups):
            in_group = valid_image & (image_optics == group)
            group_img_shells, _ = weighted_image_power_from_shells(
                operands.image_power_shells,
                jnp.where(in_group, support_mass, jnp.zeros((), support_mass.dtype)),
                operands.relion_norm_high_shell,
                in_group,
                norm_unweighted_shell_cutoff=config.norm_unweighted_shell_cutoff,
                include_unweighted_high_shell=config.include_unweighted_high_shell,
                deterministic_norm_reduction=config.deterministic_norm_reduction,
            )
            group_residual, group_power = _replace_low_shell_noise_with_relion_wavg_direct_residual_jnp(
                jnp.asarray(operands.block_noise_shells[group], dtype=jnp.float64),
                group_img_shells.astype(jnp.float64),
                jnp.where(in_group[:, None], operands.wavg_triplet_pixels[:, :, 2], jnp.float32(0.0)),
                tables.wavg_shell_indices,
                exclusive_shell_stop=int(config.direct_noise_exclusive_shell_stop),
                shell_count=n_shells,
            )
            residual_per_group.append(group_residual)
            power_per_group.append(group_power)
        residual_shells = jnp.stack(residual_per_group)
        img_power_shells = jnp.stack(power_per_group)
    wsum_sigma2_noise = stats.wsum_sigma2_noise + residual_shells
    wsum_img_power = stats.wsum_img_power + img_power_shells

    # --- 7. per-image norm residual ---------------------------------------
    block_norm_residual = operands.a2_per_image - 2.0 * operands.xa_per_image
    norm_correction = norm_correction.at[image_slot].add(
        jnp.where(valid_image, block_norm_residual, 0.0).astype(jnp.float64),
        mode="drop",
    )

    # --- 8. group scale sufficient statistics ------------------------------
    scale_xa = stats.scale_xa
    scale_aa = stats.scale_aa
    if config.accumulate_scale:
        if operands.scale_xa_per_image is not None:
            # K>1: already summed class by class under each class's own mask.
            scale_xa_per_image = operands.scale_xa_per_image
            scale_aa_per_image = operands.scale_aa_per_image
        else:
            mask_rect = jnp.asarray(tables.wavg_scale_pixel_mask, dtype=bool).reshape(1, -1)
            zero_f32 = jnp.float32(0.0)
            scale_xa_per_image = jnp.sum(
                jnp.where(mask_rect, operands.wavg_triplet_pixels[:, :, 0], zero_f32).astype(
                    jnp.float64
                ),
                axis=1,
            )
            scale_aa_per_image = jnp.sum(
                jnp.where(mask_rect, operands.wavg_triplet_pixels[:, :, 1], zero_f32).astype(
                    jnp.float64
                ),
                axis=1,
            )
        group_slot = _drop_index(operands.group_ids, int(config.n_scale_groups))
        keep_group = valid_image & (jnp.asarray(operands.group_ids, dtype=jnp.int32) >= 0)
        scale_xa = scale_xa.at[group_slot].add(
            jnp.where(keep_group, scale_xa_per_image, 0.0).astype(jnp.float64), mode="drop"
        )
        scale_aa = scale_aa.at[group_slot].add(
            jnp.where(keep_group, scale_aa_per_image, 0.0).astype(jnp.float64), mode="drop"
        )

    # --- 9. rotation posterior sums ----------------------------------------
    probs_sum_t = jnp.sum(probs, axis=-1)
    coarse_slot = _drop_index(operands.row_coarse_rot, int(config.n_coarse_rot))
    rotation_posterior_sums = stats.rotation_posterior_sums.at[coarse_slot].add(
        probs_sum_t.astype(jnp.float64), mode="drop"
    )

    # --- 10. per-image score and pose fields -------------------------------
    # ``_relion_cuda_fine_log_evidence_offset`` is exactly ``-min_diff2``.
    log_score_offset = (-jnp.asarray(operands.min_diff2)).astype(jnp.float64)
    class_log_z = jnp.asarray(operands.class_log_z, dtype=jnp.float64)
    best_log_score_chunk = jnp.asarray(operands.best_log_score, dtype=jnp.float64)
    finite = jnp.isfinite(best_log_score_chunk)
    neg_inf = jnp.asarray(-jnp.inf, dtype=jnp.float64)
    absolute = class_log_z + log_score_offset

    log_evidence = stats.log_evidence.at[image_slot].set(
        jnp.where(finite, absolute, neg_inf), mode="drop"
    )
    score_log_z = stats.score_log_z.at[image_slot].set(
        jnp.where(finite, absolute, neg_inf), mode="drop"
    )
    best_log_score = stats.best_log_score.at[image_slot].set(
        best_log_score_chunk + log_score_offset, mode="drop"
    )
    max_posterior = stats.max_posterior.at[image_slot].set(
        jnp.asarray(operands.max_posterior, dtype=stats.max_posterior.dtype), mode="drop"
    )

    best_cell_index = jnp.asarray(operands.best_cell_index, dtype=jnp.int64)
    best_local_rot = (best_cell_index // jnp.int64(n_fine_trans)).astype(jnp.int32)
    best_translation = best_cell_index % jnp.int64(n_fine_trans)
    best_cell_values = (
        jnp.asarray(operands.best_fine_rot, dtype=jnp.int64) * jnp.int64(n_fine_trans)
        + best_translation
    )
    best_cell = stats.best_cell.at[image_slot].set(best_cell_values, mode="drop")
    best_local_rot_out = stats.best_local_rot.at[image_slot].set(best_local_rot, mode="drop")

    # --- 11. the class axis (K>1) ------------------------------------------
    # Each class's evidence and winner in the image's absolute coordinates, and
    # its pruned M-step mass (thr_wsum_pdf_class, ml_optimiser.cpp:10497).
    classes = stats.classes
    if classes is not None:
        n_classes = int(config.n_classes)
        class_best = jnp.asarray(operands.per_class_best_log_score, dtype=jnp.float64).reshape(
            image_capacity, n_classes
        )
        class_absolute = (
            jnp.asarray(operands.per_class_log_z, dtype=jnp.float64).reshape(image_capacity, n_classes)
            + log_score_offset[:, None]
        )
        class_mass = jax.ops.segment_sum(
            probs_sum_t.astype(jnp.float64),
            jnp.asarray(operands.row_class, dtype=jnp.int32),
            num_segments=n_classes,
        )
        classes = classes._replace(
            log_evidence=classes.log_evidence.at[image_slot].set(
                jnp.where(jnp.isfinite(class_best), class_absolute, neg_inf), mode="drop"
            ),
            best_log_score=classes.best_log_score.at[image_slot].set(
                class_best + log_score_offset[:, None], mode="drop"
            ),
            best_cell=classes.best_cell.at[image_slot].set(
                jnp.asarray(operands.per_class_best_cell, dtype=jnp.int64).reshape(image_capacity, n_classes),
                mode="drop",
            ),
            posterior_sums=classes.posterior_sums + class_mass,
        )

    return ResidentStatistics(
        wsum_sigma2_noise=wsum_sigma2_noise,
        wsum_img_power=wsum_img_power,
        sigma2_offset=sigma2_offset,
        sumw=sumw,
        norm_correction=norm_correction,
        scale_xa=scale_xa,
        scale_aa=scale_aa,
        rotation_posterior_sums=rotation_posterior_sums,
        log_evidence=log_evidence,
        best_log_score=best_log_score,
        max_posterior=max_posterior,
        best_cell=best_cell,
        score_log_z=score_log_z,
        best_local_rot=best_local_rot_out,
        invalid_best_rows=stats.invalid_best_rows,
        classes=classes,
    )


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
) -> _CoarseNormalizationReuse | None:
    """The retained coarse normalization of a zero-oversampling pass, on the device.

    ``None`` unless the caller supplied the coarse float32 sum; the production
    gate requires the winner and Pmax with it.
    """

    from relax.sparse_pass2.resident_candidates import coarse_winner_cells

    if relion_f32_normalization_sum_weight is None:
        return None
    n_images = int(tables.n_images)
    sum_weight = np.asarray(relion_f32_normalization_sum_weight, dtype=np.float64).reshape(-1)
    if sum_weight.shape != (n_images,):
        raise ValueError(
            f"relion_f32_normalization_sum_weight must have shape ({n_images},), got {sum_weight.shape}"
        )
    max_posterior = np.asarray(relion_coarse_max_posterior, dtype=np.float64).reshape(-1)
    if (
        max_posterior.shape != (n_images,)
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
    )
    # Stored at the image capacity like every per-image device array, so the
    # chunk programs that read them are not keyed on the subset size. Chunks
    # only read real image ids.
    padding = resident_image_capacity(n_images) - n_images
    return _CoarseNormalizationReuse(
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
):
    """One class's per-image hypotheses and candidate table (T5), exactly the K=1 build.

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
    )
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


class ResidentClassInputs(NamedTuple):
    """The class axis of a K-class pass (RELION Class3D), one entry per class.

    ``rotation_log_priors`` are each class's coarse rotation log prior with its
    ``log pdf_class`` folded in, as the compact engine takes them
    (k_class.py ``_class_rotation_prior``).
    """

    significant_sample_indices: tuple
    rotation_log_priors: tuple


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
    normalization_other_score_log_z=None,
    normalization_score_mode=None,
    return_score_log_z=False,
    return_score_log_z_only=False,
    disable_adjoint_y=False,
    disable_adjoint_ctf=False,
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
    relion_translation_angle_scale: float = 1.0,
    optics_group_ids=None,
    reconstruction_volume_current_size=None,
    reconstruction_image_radius=None,
    reconstruction_group_ids=None,
    reconstruction_group_count=None,
    classes: ResidentClassInputs | None = None,
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
    (:mod:`relax.helpers.optics_noise`), and the noise statistics come back per group.

    ``reconstruction_volume_current_size`` is the backprojector's model size when the
    images are on another grid than the reference (an optics group with another box
    or pixel size): the image-side windows keep ``reconstruction_current_size`` in image
    pixels, the accumulator and its adjoint radius use the reference model size.
    """

    from recovar import cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels
    from relax.sampling import (
        get_oversampled_translation_grid,
        infer_translation_step,
        rotation_grid_size,
    )
    from relax.sparse_pass2.sparse_pass2_window import (
        _fine_translation_prior_2d,
        _pass2_projection_budget,
        _pass2_relion_flags,
    )
    from relax.symmetry import canonicalize_rotational_symmetry

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
        # RELION's Class3D E-step; the zero-oversampling reuse and the
        # --firstiter_cc winner are the auto-refine K=1 iterations.
        _require(not firstiter_cc, "the K-class resident pass scores the Gaussian likelihood")
        _require(
            relion_f32_normalization_sum_weight is None,
            "the K-class resident pass has no zero-oversampling coarse reuse",
        )
        _require(optics_group_ids is None, "the K-class resident pass has one optics group")

    n_images = experiment_dataset.n_units
    n_coarse_trans = int(np.asarray(translations).shape[0])
    symmetry_label = canonicalize_rotational_symmetry(symmetry_label)
    # The coarse grid is RELION's asymmetric-unit HEALPix sampling
    # (healpix_sampling.cpp removeSymmetryEquivalentPoints); the caller's fine
    # rotation override already holds its children.
    n_coarse_rot = rotation_grid_size(nside_level, symmetry_label)
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape

    if current_size is None:
        # The resident drivers score RELION's window at every size, the box included
        # (window_at_box below), so the full box is an explicit current size here.
        current_size = int(experiment_dataset.image_shape[0])
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
            reference_sphere_clip=reconstruction_image_radius is not None,
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
    )
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
    require_resident_production_configuration(
        relion_x_half_mstep=relion_x_half_mstep,
        relion_exact_fine_gaussian=relion_exact_fine_gaussian,
        relion_exact_fine_normalized_cc=relion_exact_fine_normalized_cc,
        relion_firstiter_score_mode=relion_firstiter_score_mode,
        use_float64_scoring=use_float64_scoring,
        relion_firstiter_winner_take_all=relion_firstiter_winner_take_all,
        disable_adjoint_y=disable_adjoint_y,
        disable_adjoint_ctf=disable_adjoint_ctf,
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
    _require(
        bool(use_relion_f32_fine_posterior) or firstiter_cc,
        "the RELION float32 fine posterior is the segmented kernel's contract "
        "(the --firstiter_cc pass takes the winner instead)",
    )
    _require(
        bool(relion_fine_mstep_prune) or bool(relion_x_half_mstep),
        "the resident M-step reconstructs from RELION's pruned fine weights",
    )
    _require(
        jax.default_backend() == "gpu" and cuda_backproject.custom_cuda_requested(),
        "every resident stage is a CUDA FFI target",
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
    mstep_max_r = mstep_adjoint_max_r(volume_current_size, reconstruction_image_radius, reconstruction_padding_factor)
    recon_y_accum_dtype, recon_ctf_accum_dtype = relion_x_half_mstep_accumulator_dtypes(
        experiment_dataset.dtype,
        use_relion_x_half_mstep=True,
    )
    logger.info(
        "Resident pass-2 RELION x-half current-size BPref accumulator shape: "
        "volume_shape=%s score_current_size=%s model_current_size=%s padding_factor=%s "
        "recon_volume_shape=%s half_accum_shape=%s voxels=%d",
        tuple(volume_shape),
        current_size,
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
            # RELION projects through a float32 texture (AccProjector::setMdlData),
            # and so does the compact engine (dispatch's persistent texture, or the
            # same narrowing in its own block path). Left complex128, this driver's
            # projections fell back to the vmapped JAX projector: different values
            # (os1 iteration 1: hard assignments 97.7% equal, 14384091) and twice
            # the iteration time. One projection path in both engines.
            if not use_float64_scoring and relion_projector_half.dtype == jnp.complex128:
                relion_projector_half = relion_projector_half.astype(jnp.complex64)
            class_projector_halves.append(relion_projector_half)
        del stacked_halves, relion_projector_half
    class_volumes = [volume] if classes is None else [volume[k] for k in range(n_classes)]
    if projection_padding_factor > 1 and not use_relion_projector:
        from relax.reconstruction.relion_functions_relion import pad_volume_for_projection

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

    # ---- per-image hypotheses and candidate tables, one per class ---------
    # A K-class table joins the classes' own tables in RELION's class-major
    # hidden space (merge_class_tables; docs/development/resident_segments.md).
    prep_s = table_s = 0.0
    tables_by_class = []
    for significant_sample_indices, rotation_log_prior in zip(class_supports, class_rotation_priors):
        class_tables, class_prep_s, class_table_s = _class_candidate_tables(
            significant_sample_indices,
            rotation_log_prior,
            n_images=n_images,
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
        )
        tables_by_class.append(class_tables)
        prep_s += class_prep_s
        table_s += class_table_s
    table_t0 = time.time()
    tables = tables_by_class[0] if classes is None else merge_class_tables(tables_by_class)
    tables = _with_reconstruction_groups(
        tables, reconstruction_group_ids, reconstruction_group_count, n_images=n_images
    )
    table_s += time.time() - table_t0
    coarse_reuse = _coarse_normalization_reuse(
        tables,
        relion_f32_normalization_sum_weight=relion_f32_normalization_sum_weight,
        relion_coarse_hard_assignment=relion_coarse_hard_assignment,
        relion_coarse_max_posterior=relion_coarse_max_posterior,
        fine_rotation_parent=fine_rotation_parent_override,
        fine_translation_parent=fine_translation_parent,
        max_posterior_dtype=precision_policy.score_real_dtype,
    )

    # ---- window / weights / lookups (unchanged) ---------------------------
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
    )
    relion_score_full_to_compact = jnp.asarray(
        _relion_cuda_fine_full_to_compact_lookup(image_shape, current_size, window_indices_np),
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
    relion_score_translation_angles = _relion_cuda_score_translation_angles_if_available(
        fine_translations_source,
        image_shape,
        enabled=True,
        dtype=np.float64 if use_float64_scoring else np.float32,
        angle_scale=relion_translation_angle_scale,
    )
    if relion_score_translation_angles is None:
        raise ValueError("the resident scoring stage requires RELION translation angles")
    translation_phases_half = (
        None if windowed_prepare else half_translation_phase_table(fine_translations, image_shape)
    )

    n_shells = image_shape[0] // 2 + 1
    shell_indices_half = mask_relion_noise_shell_indices_to_current_window(
        make_relion_noise_shell_indices_half(image_shape),
        image_shape,
        current_size,
        window_indices,
    )
    shell_indices_noise = window_spec.recon_values(shell_indices_half)
    noise_variance_for_noise = window_spec.recon_values(noise_variance_half)
    relion_wavg_rectangle = _make_relion_wavg_rectangle(
        image_shape,
        current_size,
        recon_window_indices,
        reconstruction_current_size=mstep_current_size,
    )
    n_rect = int(relion_wavg_rectangle.centered_indices.size)
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
    n_fine_rot = int(np.asarray(fine_rotations_override).shape[0])
    # K>1 caches every class's fine grid, class k at k * n_fine_rot.
    n_projections = n_classes * n_fine_rot
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
    max_projection_cache_bytes = _projection_cache_max_bytes_for_pass(device_memory_bytes)
    projection_kwargs = _projection_kwargs_for_relion_score_window(
        window_spec.projection_kwargs(return_abs2=False),
        use_relion_projector=use_relion_projector,
        current_size=current_size,
    )
    projection_kwargs["mask_current_image_disk"] = bool(projection_mask_current_image_disk)

    fine_grid = jnp.asarray(fine_rotations_override, dtype=precision_policy.score_real_dtype)

    def project_fine_rotations(rotations, class_index=0):
        """(score, recon, |recon|^2) projections of ``rotations``, as the cache holds them."""

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
            relion_projector_half=class_projector_halves[class_index],
            relion_projector_r_max=relion_projector_r_max,
            projection_padding_factor=projection_padding_factor,
            **projection_kwargs,
        )
        recon, recon_abs2 = precision_policy.cast_local_noise_projection_scores(recon, recon_abs2)
        if relion_native_fine_units:
            # Score rows only, for the cached and the streamed paths alike; the
            # recon rows feed the M-step and noise sums in RECOVAR units.
            score = _relion_native_fine_units_in_place(score, native_fft_size)
        return score, recon, recon_abs2

    def project_ids(ids):
        """Projections of the host projection ids ``class * n_fine_rot + rotation``, in order.

        With K>1 each class projects its own ids from its own reference; a
        class's call is padded to a whole number of stream quanta so the
        projection programs see few distinct lengths, as the K=1 stream does.
        """

        ids = np.asarray(ids, dtype=np.int64)
        if n_classes == 1:
            return project_fine_rotations(fine_grid[jnp.asarray(ids, dtype=jnp.int32)])
        klass = ids // n_fine_rot
        order = np.argsort(klass, kind="stable")
        parts = []
        for class_index in np.unique(klass):
            class_ids = ids[order[klass[order] == class_index]] % n_fine_rot
            n_call = -(-class_ids.size // _STREAM_SLOT_QUANTUM) * _STREAM_SLOT_QUANTUM
            padded = np.full(n_call, class_ids[0], dtype=np.int64)
            padded[: class_ids.size] = class_ids
            projected = project_fine_rotations(
                fine_grid[jnp.asarray(padded, dtype=jnp.int32)], int(class_index)
            )
            parts.append(tuple(values[: class_ids.size] for values in projected))
        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.size)
        inverse_device = jnp.asarray(inverse, dtype=jnp.int32)
        return tuple(
            jnp.concatenate([part[field] for part in parts], axis=0)[inverse_device]
            for field in range(3)
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
    if _resident_operands_requested() and not firstiter_cc:
        operand_bytes, operand_peak_bytes = _resident_half_operand_sizes(
            n_images=n_images,
            n_windowed=n_windowed,
            n_recon_windowed=n_recon_windowed,
            n_rect=n_rect,
            n_shells=n_shells,
            n_fine_trans=n_fine_trans,
            precision_policy=precision_policy,
            resolved_spectrum_norm=resolved_spectrum_norm,
        )
        if _resident_operands_fit(
            operand_peak_bytes,
            device_available_bytes(physical_free_bytes, allocator_free_bytes, pool_free_bytes),
        ):
            reserved_operand_bytes = operand_bytes
    stream_projection_budget_bytes = _stream_projection_budget_bytes(
        max_projection_cache_bytes,
        physical_free_bytes=physical_free_bytes,
        allocator_free_bytes=allocator_free_bytes,
        pool_free_bytes=pool_free_bytes,
        reserved_bytes=reserved_operand_bytes,
    )
    stream_projections = not _projection_cache_fits_budget(
        transient_projection_bytes, stream_projection_budget_bytes
    )
    if stream_projections:
        score_cache = recon_cache = recon_abs2_cache = None
        logger.info(
            "Resident pass-2 projections are streamed per chunk: the %d-rotation cache "
            "would take %.2f GiB against a %.2f GiB budget (cache share %.2f GiB) "
            "at %.1f KiB per rotation (physical free %s, allocator free %s, "
            "pool free %s, reserved operands %.2f GiB)",
            n_projections,
            transient_projection_bytes / float(1024**3),
            stream_projection_budget_bytes / float(1024**3),
            max_projection_cache_bytes / float(1024**3),
            projection_bytes_per_rotation / 1024.0,
            "unknown" if physical_free_bytes is None else f"{physical_free_bytes / float(1024**3):.2f} GiB",
            "unknown" if allocator_free_bytes is None else f"{allocator_free_bytes / float(1024**3):.2f} GiB",
            "unknown" if pool_free_bytes is None else f"{pool_free_bytes / float(1024**3):.2f} GiB",
            reserved_operand_bytes / float(1024**3),
        )
    else:
        cache_t0 = time.time()
        if n_classes == 1:
            score_cache, recon_cache, recon_abs2_cache = project_fine_rotations(
                fine_rotations_override
            )
        else:
            class_caches = [
                project_fine_rotations(fine_rotations_override, class_index)
                for class_index in range(n_classes)
            ]
            score_cache, recon_cache, recon_abs2_cache = (
                jnp.concatenate([cache[field] for cache in class_caches], axis=0) for field in range(3)
            )
            del class_caches
        logger.info(
            "Resident pass-2 projection cache: cached %d fine rotations in %.2fs "
            "(estimated transient %.2f GiB)",
            n_projections,
            time.time() - cache_t0,
            transient_projection_bytes / float(1024**3),
        )

    # ---- capacity plan ----------------------------------------------------
    row_ladder = parse_env_capacity_ladder(_ROW_CAPACITY_LADDER_ENV, _DEFAULT_ROW_CAPACITY_LADDER)
    if stream_projections:
        row_ladder = _stream_row_capacity_ladder(
            row_ladder,
            bytes_per_rotation=projection_bytes_per_rotation,
            max_projection_bytes=stream_projection_budget_bytes,
        )
    else:
        # The chunk gathers its rows out of the cache into one [rows, pixels]
        # block; at box 800 (10202, current size 304) capacity 131072 needs
        # 35.7 GiB for it (14434683). The budget is read again now that the
        # cache is resident; unknown readings do not cap.
        gather_budget_bytes = _stream_projection_budget_bytes(
            device_memory_bytes if device_memory_bytes is not None else 1 << 62,
            physical_free_bytes=_device_free_memory_bytes(),
            allocator_free_bytes=_jax_allocator_free_memory_bytes(),
            pool_free_bytes=_jax_allocator_pool_free_bytes(),
            reserved_bytes=reserved_operand_bytes,
        )
        row_ladder = _cached_row_capacity_ladder(
            row_ladder,
            bytes_per_row=int(n_windowed) * np.dtype(precision_policy.score_complex_dtype).itemsize,
            max_gather_bytes=gather_budget_bytes,
        )
        logger.info(
            "Resident pass-2 cached-path row capacities %s: gather budget %.2f GiB at %.1f KiB per row",
            ",".join(str(v) for v in row_ladder),
            gather_budget_bytes / float(1024**3),
            int(n_windowed) * np.dtype(precision_policy.score_complex_dtype).itemsize / 1024.0,
        )
    image_ladder = _cap_image_capacity_ladder(
        parse_env_capacity_ladder(_IMAGE_CAPACITY_LADDER_ENV, _DEFAULT_IMAGE_CAPACITY_LADDER),
        n_fine_trans=n_fine_trans,
        n_recon_pixels=n_recon_windowed,
        max_tile_bytes=_max_translation_tile_bytes_for_pass(
            device_memory_bytes, has_external_normalization=False
        ),
    )
    mstep_block_rows = _resolve_mstep_block_rows(
        n_recon_pixels=n_recon_windowed,
        max_block_bytes=_max_adjoint_block_bytes_for_pass(device_memory_bytes),
        row_capacity_ladder=row_ladder,
    )
    chunks = plan_capacity_chunks(
        tables,
        row_capacity_ladder=row_ladder,
        image_capacity_ladder=image_ladder,
    )
    plan = ResidentPass2Plan(
        chunks=tuple(chunks),
        row_capacity_ladder=tuple(row_ladder),
        image_capacity_ladder=tuple(image_ladder),
        mstep_block_rows=int(mstep_block_rows),
    )
    table_s = time.time() - table_t0
    row_slots = sum(int(chunk.row_capacity) for chunk in chunks)
    image_slots = sum(int(chunk.image_capacity) for chunk in chunks)
    logger.info(
        "Resident pass-2 plan: %d images, %d candidate rows -> %d chunks "
        "(row capacities %s, image capacities %s, M-step block rows %d, "
        "row occupancy %.3f of %d slots, image occupancy %.3f of %d slots); "
        "setup hypothesis_prep=%.2fs table+plan=%.2fs",
        tables.n_images,
        tables.n_rows,
        len(chunks),
        ",".join(str(v) for v in plan.row_capacity_ladder),
        ",".join(str(v) for v in plan.image_capacity_ladder),
        plan.mstep_block_rows,
        tables.n_rows / max(row_slots, 1),
        row_slots,
        tables.n_images / max(image_slots, 1),
        image_slots,
        prep_s,
        table_s,
    )

    # ---- preparation arguments --------------------------------------------
    # One keyword set, used by whichever preparation the pass selects: the
    # once-per-half resident preparation below, or the per-chunk call that
    # stays as its oracle. The preparation is per-image pure, so the two return
    # the same rows; the resident form runs it once for the half instead of
    # once per chunk, which is where the chunk loop's launches came from.
    bucket_io_kwargs = dict(
        noise_variance_half=noise_variance_half,
        fine_translations=fine_translations,
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

    # ---- resident row-aligned tables --------------------------------------
    mstep_grid = (
        fine_grid
        if fine_mstep_rotations_override is None
        else jnp.asarray(fine_mstep_rotations_override, dtype=precision_policy.score_real_dtype)
    )
    coarse_parent_np = np.asarray(fine_rotation_parent_override, dtype=np.int32)
    cached_slot_fine_rot = None
    if n_classes > 1:
        # Indexed by projection id: the M-step rotation is the class's fine
        # rotation, and the rotation-mass slot is (class, coarse rotation).
        mstep_grid = jnp.tile(mstep_grid, (n_classes, 1, 1))
        coarse_parent_np = (
            np.arange(n_classes, dtype=np.int32)[:, None] * np.int32(n_coarse_rot) + coarse_parent_np[None, :]
        ).reshape(-1)
        if not stream_projections:
            cached_slot_fine_rot = jnp.asarray(np.tile(np.arange(n_fine_rot, dtype=np.int32), n_classes))
    coarse_parent_grid = jnp.asarray(coarse_parent_np, dtype=jnp.int32)
    projection_score_cache = None if score_cache is None else jnp.asarray(score_cache)
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
        n_images=n_images,
        n_coarse_rot=n_classes * n_coarse_rot,
        n_scale_groups=n_scale_groups,
        current_size=current_size,
        include_unweighted_high_shell=include_unweighted_norm_high_shell,
        use_exact_relion_gaussian=use_exact_relion_gaussian,
        relion_wavg_atomic_direct_noise=relion_wavg_atomic_direct_noise,
        relion_wavg_atomic_scale_aa=relion_wavg_atomic_scale_aa,
        accumulate_scale=scale_groups_available,
        source_faithful_spectrum_norm=resolved_spectrum_norm,
        n_optics_groups=n_optics_groups,
        n_classes=n_classes,
    )
    stats = make_resident_statistics(
        stats_config, max_posterior_dtype=precision_policy.score_real_dtype
    )
    image_tables = _ChunkImageTables(
        shell_indices_half=jnp.asarray(shell_indices_half, dtype=jnp.int32),
        wavg_shell_indices=jnp.asarray(relion_wavg_rectangle.shell_indices, dtype=jnp.int32),
        wavg_scale_pixel_mask=jnp.asarray(scale_pixel_mask_rect_np, dtype=bool),
        translation_sqdist_ang=None,
    )

    # One x-half BPref pair per accumulator slot: RELION's BPref[iclass], and for
    # VDAM's pseudo-halfsets BPref[iclass + half * nr_classes].
    n_slots = int(tables.n_slots)
    Ft_y_total = tuple(jnp.zeros(recon_volume_size, dtype=recon_y_accum_dtype) for _ in range(n_slots))
    Ft_ctf_total = tuple(jnp.zeros(recon_volume_size, dtype=recon_ctf_accum_dtype) for _ in range(n_slots))
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
    # The --firstiter_cc iteration scores the compact engine's translated
    # normalized-CC tiles, which only the per-chunk preparation builds; it is
    # one iteration at a small current size.
    if _resident_operands_requested() and not firstiter_cc:
        operand_bytes, operand_peak_bytes = _resident_half_operand_sizes(
            n_images=n_images,
            n_windowed=n_windowed,
            n_recon_windowed=n_recon_windowed,
            n_rect=n_rect,
            n_shells=n_shells,
            n_fine_trans=n_fine_trans,
            precision_policy=precision_policy,
            resolved_spectrum_norm=resolved_spectrum_norm,
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
        if not _resident_operands_fit(operand_peak_bytes, available_bytes):
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
                            current_size=current_size,
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
                        )
                        warmup = _submit_resident_chunk_warmup(
                            warm_pool,
                            chunks=chunks,
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
                                current_size=current_size,
                                mstep_current_size=volume_current_size,
                                mstep_max_r=mstep_max_r,
                                image_shape=image_shape,
                                recon_volume_shape=recon_volume_shape,
                                max_adjoint_block_bytes=max_adjoint_block_bytes,
                                stats_config=stats_config,
                                use_rfloat_ctf_wavg=presence.has_direct_ctf_rfloat,
                                use_translate_sum_kernel=True,
                                bpref_recon_operand=presence.has_recon_weight,
                                reuse_coarse_normalization=coarse_reuse is not None,
                                n_slots=int(tables.n_slots),
                                mstep_subtract_ctf_projection=bool(mstep_subtract_ctf_projection),
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
                try:
                    resident_operands = prepare_resident_half_operands(
                        experiment_dataset,
                        np.arange(n_images, dtype=np.int64),
                        bucket_io_kwargs=bucket_io_kwargs,
                        window_indices=window_indices,
                        recon_window_indices=recon_window_indices,
                        wavg_rect_indices=relion_wavg_rectangle.centered_indices,
                        noise_shell_indices_half=shell_indices_half,
                        n_noise_shells=int(n_shells),
                        image_shape=image_shape,
                        current_size=current_size,
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
                    )
                except ResidentOperandsUnsupported as reason:
                    logger.info(
                        "Resident pass-2 keeps the per-chunk operand preparation: %s", reason
                    )
                    resident_operands = None
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

    verify_operands = resident_operands is not None and _resident_operands_verify_enabled()

    # ---- chunk loop --------------------------------------------------------
    # With the warm-up on, collect the (program, spec) keys the loop actually
    # submits. Comparing them against what the helper compiled is the hit rate:
    # a key the loop used and the warm-up did not is a program the loop compiled
    # itself, which is what a mis-predicted spec looks like.
    submitted_keys = set() if warmup is not None else None
    loop_t0 = time.time()
    for chunk in chunks:
        Ft_y_total, Ft_ctf_total, stats = _run_resident_chunk(
            chunk,
            tables=tables,
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
            verify_operands=verify_operands and chunk is chunks[0],
            image_shape=image_shape,
            current_size=current_size,
            mstep_current_size=volume_current_size,
            mstep_max_r=mstep_max_r,
            recon_volume_shape=recon_volume_shape,
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
            cuda_backproject=em_cuda_kernels,
            submitted_keys=submitted_keys,
            optics_groups_np=optics_groups_np,
            relion_native_fine_units=relion_native_fine_units,
            coarse_reuse=coarse_reuse,
            firstiter_cc=firstiter_cc,
            mstep_subtract_ctf_projection=bool(mstep_subtract_ctf_projection),
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
    Ft_y_out, Ft_ctf_out = [], []
    for class_Ft_y, class_Ft_ctf in zip(Ft_y_total, Ft_ctf_total):
        class_Ft_y, class_Ft_ctf = finalize_half_volume_bpref(
            class_Ft_y,
            class_Ft_ctf,
            recon_volume_shape,
            logger=logger,
            label="Resident pass-2",
            symmetry_label=symmetry_label,
            relion_x_half=True,
        )
        class_Ft_y, class_Ft_ctf = relion_x_half_accumulators_to_public_layout(
            class_Ft_y,
            class_Ft_ctf,
            recon_volume_shape,
        )
        Ft_y_out.append(class_Ft_y)
        Ft_ctf_out.append(class_Ft_ctf)

    finalized = finalize_statistics(stats, config=stats_config, n_images=n_images)
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
    VDAM's accumulator adapter (``relax.vdam.estep_common._arrays_to_accumulators``) reads.
    """

    k, n_classes = int(class_index), int(result.n_classes)
    if int(result.n_slot_groups) == 1:
        return result.Ft_y[k], result.Ft_ctf[k]
    slots = [k + n_classes * group for group in range(int(result.n_slot_groups))]
    return (
        jnp.stack([result.Ft_y[a] for a in slots], axis=0),
        jnp.stack([result.Ft_ctf[a] for a in slots], axis=0),
    )


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
    normalization_other_score_log_z=None,
    normalization_score_mode=None,
    return_score_log_z=False,
    return_score_log_z_only=False,
    disable_adjoint_y=False,
    disable_adjoint_ctf=False,
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
    relion_translation_angle_scale: float = 1.0,
    optics_group_ids=None,
    reconstruction_volume_current_size=None,
    reconstruction_image_radius=None,
    reconstruction_group_ids=None,
    reconstruction_group_count=None,
):
    """Device-resident K=1 sparse pass 2; same signature and return as the compact engine.

    A one-class :func:`_resident_pass2`. See the module docstring for what is
    layout-equal to the compact engine and what is a deliberate reduction-order
    change.
    """

    # Every parameter, forwarded by name: the signature is the compact engine's.
    result = _resident_pass2(**locals())
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
    )


def _chunk_segment_offsets(tables, chunk, n_fine_trans: int) -> np.ndarray:
    """Cell offsets of each chunk image slot, in the segmented handler's units.

    Slot ``b`` of the chunk owns the cells of image ``image_start + b``; padded
    slots and the rows past ``n_valid_rows`` are covered by no segment, which
    the handler treats exactly as an all ``-inf`` rectangular row.
    """

    image_capacity = int(chunk.image_capacity)
    n_valid_images = int(chunk.n_valid_images)
    offsets = np.full(image_capacity + 1, chunk.n_valid_rows * int(n_fine_trans), dtype=np.int64)
    starts = (
        np.asarray(
            tables.row_offsets[chunk.image_start : chunk.image_start + n_valid_images + 1],
            dtype=np.int64,
        )
        - int(chunk.row_start)
    ) * int(n_fine_trans)
    offsets[: n_valid_images + 1] = starts
    if int(offsets[-1]) > int(chunk.row_capacity) * int(n_fine_trans):
        raise ValueError("chunk segment offsets exceed the chunk's cell capacity")
    return offsets.astype(np.int32)


class _Placement(NamedTuple):
    """How a chunk constructor turns host values into program inputs.

    The chunk loop places them on the device; the compile-ahead warm-up places
    their shape and dtype only. Having one constructor with two placements
    rather than two constructors means a warm-up cannot describe a program the
    loop does not run, which is the failure a hit rate would only report after
    the compile time had already been spent.
    """

    array: object
    scalar: object


_PLACE_ON_DEVICE = _Placement(
    array=lambda value, dtype: jnp.asarray(value, dtype=dtype),
    # ``np.asarray`` first: a NumPy *scalar* reaches the device through one
    # eager ``convert_element_type`` per chunk, a 0-d NumPy *array* of the same
    # dtype through a plain transfer. Same dtype, shape, weak type and value
    # either way.
    scalar=lambda value, dtype: _scalar_operand(value, dtype),
)

_PLACE_AS_AVAL = _Placement(
    array=lambda value, dtype: jax.ShapeDtypeStruct(np.shape(value), jnp.dtype(dtype)),
    scalar=lambda value, dtype: jax.ShapeDtypeStruct((), jnp.dtype(dtype)),
)


_STREAM_SLOT_QUANTUM = 8192
# Share of the measured free device memory a chunk-local projection cache may
# take; the rest stays for the chunk's scoring and M-step working set, which
# have their own device-fraction budgets.
_STREAM_FREE_MEMORY_FRACTION = 0.5
# While a chunk's projections are padded to the row capacity, the projected
# block and its padded copy are both live (flipqual 10097 it13: a 12.16 GiB pad
# at row capacity 131072 ran the device out of memory, 14397073).
_STREAM_PEAK_COPIES = 2


def _resident_half_operand_sizes(
    *,
    n_images,
    n_windowed,
    n_recon_windowed,
    n_rect,
    n_shells,
    n_fine_trans,
    precision_policy,
    resolved_spectrum_norm,
):
    """``(bytes, peak bytes)`` of one half's resident operands.

    The preparation holds the operands plus one reordered copy of its largest
    array (resident_operands.stack), hence the peak.
    """

    _, norm_high_shell_dtype = relion_powerclass_noise_dtypes(
        real_dtype=precision_policy.score_real_dtype,
        source_faithful_spectrum_norm=resolved_spectrum_norm,
    )
    operand_bytes = resident_half_operand_bytes(
        n_images=int(n_images),
        n_score_pixels=int(n_windowed),
        n_recon_pixels=int(n_recon_windowed),
        n_rect_pixels=int(n_rect),
        n_noise_shells=int(n_shells),
        n_fine_trans=int(n_fine_trans),
        score_complex_bytes=np.dtype(precision_policy.score_complex_dtype).itemsize,
        real_bytes=np.dtype(precision_policy.score_real_dtype).itemsize,
        norm_high_shell_bytes=np.dtype(norm_high_shell_dtype).itemsize,
    )
    peak_bytes = operand_bytes + resident_image_capacity(int(n_images)) * max(
        int(n_windowed), int(n_recon_windowed), int(n_rect)
    ) * np.dtype(precision_policy.score_complex_dtype).itemsize
    return operand_bytes, peak_bytes


def _resident_operands_fit(operand_peak_bytes, available_bytes) -> bool:
    """Whether one half's resident operands are admitted at this reading.

    The one predicate for both the reservation before the projection cache
    decision and the admission after it.
    """

    return int(operand_peak_bytes) <= resident_operands_max_bytes(available_bytes)


def _stream_projection_budget_bytes(
    max_projection_cache_bytes,
    *,
    physical_free_bytes,
    allocator_free_bytes,
    pool_free_bytes=None,
    reserved_bytes=0,
):
    """Chunk-local projection budget: the cache share, capped by measured free memory.

    The readings are taken when the pass plans its chunks, so they see whatever
    earlier passes and iterations left resident; the per-iteration cache share
    alone would ignore that. What the allocator can still hand out is
    :func:`~relax.sparse_pass2.sparse_pass2_budget.device_available_bytes`; an
    unknown reading does not cap. ``reserved_bytes`` (the half's resident
    operands, allocated after the reading) comes off first; half of the rest
    stays for the half's accumulators and the chunk working set, which have
    their own budgets.
    """

    available = device_available_bytes(physical_free_bytes, allocator_free_bytes, pool_free_bytes)
    budget = int(max_projection_cache_bytes)
    if available is not None:
        usable = max(0.0, available - float(reserved_bytes))
        budget = min(budget, int(usable * _STREAM_FREE_MEMORY_FRACTION))
    return max(0, budget)


def _stream_row_capacity_ladder(row_ladder, *, bytes_per_rotation, max_projection_bytes):
    """Row capacities whose chunk-local projection cache fits the cache budget.

    A streamed chunk projects at most one rotation per row, so capacity times
    the per-rotation bytes bounds its cache; the budget is the one the
    per-iteration cache failed, which keeps the pass's peak where it would
    have been had that cache fitted.
    """

    kept = tuple(
        int(c)
        for c in row_ladder
        if _STREAM_PEAK_COPIES * float(c) * float(bytes_per_rotation) <= float(max_projection_bytes)
    )
    if not kept:
        # A configuration refusal, so the resident default falls back to the
        # compact engine with a logged reason and an explicit =1 stops the run
        # (dispatch._resident_with_compact_default catches only this type).
        raise ResidentConfigurationUnsupported(
            "the device-resident sparse pass 2 does not fit this pass: even the smallest row capacity "
            f"{min(int(c) for c in row_ladder)} needs "
            f"{_STREAM_PEAK_COPIES * min(int(c) for c in row_ladder) * bytes_per_rotation / float(1024 ** 3):.2f} GiB of "
            f"streamed projections against a {max_projection_bytes / float(1024 ** 3):.2f} GiB budget"
        )
    return kept


def _cached_row_capacity_ladder(row_ladder, *, bytes_per_row, max_gather_bytes):
    """Row capacities whose gathered chunk of cached score projections fits the budget.

    With the whole fine grid cached, a chunk still gathers one score projection
    per row (``score_resident_chunk``: ``projection_score_cache[row_fine_rot]``),
    so capacity times the score-row bytes is one live block. RELION projects each
    orientation inside its diff2 kernel and never holds such a block; bounding the
    chunk by measured free memory keeps the gather within the device. The budget
    is :func:`_stream_projection_budget_bytes` with the device as its cap, read
    after the cache is built. A refusal falls back to the compact engine by
    default, as :func:`_stream_row_capacity_ladder` does.
    """

    kept = tuple(int(c) for c in row_ladder if float(c) * float(bytes_per_row) <= float(max_gather_bytes))
    if not kept:
        smallest = min(int(c) for c in row_ladder)
        raise ResidentConfigurationUnsupported(
            f"The device-resident K=1 sparse pass 2 ({RESIDENT_PASS2_ENV}=1) does not implement "
            f"this configuration: even the smallest row capacity {smallest} gathers "
            f"{smallest * bytes_per_row / float(1024 ** 3):.2f} GiB of cached projections against a "
            f"{max_gather_bytes / float(1024 ** 3):.2f} GiB budget. Clear the flag to use the compact "
            "engine; this path never falls back silently."
        )
    return kept


def _stream_slot_count(n_unique: int, row_capacity: int) -> int:
    """Projection-call length for ``n_unique`` rotations: a quantum multiple, capped."""

    quantum = _STREAM_SLOT_QUANTUM
    return int(min(int(row_capacity), max(quantum, -(-int(n_unique) // quantum) * quantum)))


def _stream_chunk_projections(
    rows,
    host_row_fine_rot,
    *,
    n_valid_rows,
    row_capacity,
    project,
    n_fine_rot,
    mstep_grid,
    coarse_parent_grid,
):
    """Project one chunk's distinct fine rotations and re-index its rows to them.

    ``host_row_fine_rot`` holds the rows' projection ids (the fine rotation, or
    ``class * n_fine_rot + rotation`` with K>1 classes) and ``project`` maps
    projection ids to the three projection arrays.

    Returns ``(rows, slot_fine_rot, caches, mstep_grid, coarse_parent_grid)``
    where ``rows.row_fine_rot`` now holds each row's chunk-local cache slot,
    ``slot_fine_rot`` [row_capacity] maps a slot back to its global fine
    rotation id, and the caches and the two grids are indexed by slot. Every
    consumer gathers by slot exactly as it gathered by id from the
    per-iteration caches, so the gathered projections are the same arrays; the
    winning rotation is mapped back through ``slot_fine_rot``.

    The cache arrays are always ``row_capacity`` long, so the chunk programs
    stay keyed on the capacity class alone; only the projection call is
    quantised to the chunk's distinct-rotation count. Valid rows read only the
    first ``n_unique`` slots; padded rows read slot 0, a real rotation, and
    carry a zero posterior. Slots past the projection call are zero and unread.
    """

    valid = np.asarray(host_row_fine_rot[: int(n_valid_rows)], dtype=np.int64)
    unique, inverse = np.unique(valid, return_inverse=True)
    if unique.size == 0:
        unique = np.zeros(1, dtype=np.int64)
    n_project = _stream_slot_count(unique.size, row_capacity)
    slot_fine_rot = np.full(int(row_capacity), unique[0], dtype=np.int64)
    slot_fine_rot[: unique.size] = unique
    row_slot = np.zeros(int(row_capacity), dtype=np.int32)
    row_slot[: valid.size] = inverse.astype(np.int32, copy=False)

    slot_ids_device = jnp.asarray(slot_fine_rot, dtype=jnp.int32)
    projected = project(slot_fine_rot[:n_project])
    pad = int(row_capacity) - n_project

    def full_length(values):
        if pad == 0:
            return values
        widths = [(0, pad)] + [(0, 0)] * (values.ndim - 1)
        return jnp.pad(values, widths)

    caches = tuple(full_length(values) for values in projected)
    return (
        rows._replace(row_fine_rot=jnp.asarray(row_slot, dtype=jnp.int32)),
        jnp.asarray(slot_fine_rot % int(n_fine_rot), dtype=jnp.int32),
        caches,
        mstep_grid[slot_ids_device],
        coarse_parent_grid[slot_ids_device],
    )


def _row_projection_ids(host_chunk, n_fine_rot: int | None) -> np.ndarray:
    """Each row's projection: its fine rotation, offset by ``class * n_fine_rot`` for K>1.

    The class-stacked projection caches and grids hold class ``k``'s fine grid at
    ``k * n_fine_rot``, so every consumer that gathers by a row's rotation id
    gathers its class's projection with the same statement.
    """

    row_fine_rot = np.asarray(host_chunk["row_fine_rot"], dtype=np.int64)
    if n_fine_rot is None:
        return row_fine_rot.astype(np.int32)
    return (row_fine_rot + np.asarray(host_chunk["row_class"], dtype=np.int64) * int(n_fine_rot)).astype(
        np.int32
    )


def _chunk_class_layout(host_chunk, chunk, *, n_classes: int, n_fine_trans: int, place) -> _ChunkClassLayout:
    """The (slot, class) posterior sub-segments of one chunk."""

    row_capacity = int(chunk.row_capacity)
    image_capacity = int(chunk.image_capacity)
    n_valid_rows = int(chunk.n_valid_rows)
    n_segments = image_capacity * int(n_classes)
    row_class = np.asarray(host_chunk["row_class"], dtype=np.int64)
    row_segment = np.full(row_capacity, n_segments, dtype=np.int64)
    row_segment[:n_valid_rows] = (
        np.asarray(host_chunk["row_image_local"][:n_valid_rows], dtype=np.int64) * int(n_classes)
        + row_class[:n_valid_rows]
    )
    if np.any(np.diff(row_segment[:n_valid_rows]) < 0):
        raise ValueError("chunk rows must be image-major, then class-major")
    segment_rows = np.bincount(row_segment[:n_valid_rows], minlength=n_segments)
    segment_row_start = np.concatenate([[0], np.cumsum(segment_rows)])
    return _ChunkClassLayout(
        row_class=place.array(row_class, jnp.int32),
        row_segment=place.array(row_segment, jnp.int32),
        segment_offsets=place.array(segment_row_start * int(n_fine_trans), jnp.int32),
        segment_row_start=place.array(segment_row_start[:-1], jnp.int64),
    )


def _chunk_mstep_layout(host_chunk, *, place) -> _ChunkMstepLayout:
    """The chunk's accumulator slot per row (docs/development/resident_segments.md).

    Slot ``a = class + K * slot_offset[unit]``: a class's rows (Class3D, RELION's
    ``BPref[iclass]``, ml_optimiser.cpp:10826) and, for VDAM, a pseudo-halfset's
    (``iclass + (part_id % 2) * nr_classes``, acc_ml_optimiser_impl.h:4800-4804).
    The slot-major order itself is taken on the device, over the live rows only
    (:func:`_make_mstep_block_inputs`).
    """

    return _ChunkMstepLayout(row_slot=place.array(host_chunk["row_slot"], jnp.int32))


def _make_chunk_row_arrays(tables, chunk, n_fine_trans, *, place, n_fine_rot=None) -> _ChunkRowArrays:
    """One chunk's row-aligned inputs, on the device or as avals.

    Everything here is host NumPy over the plan, so the aval placement costs
    only the materialize and does no device work at all. The shapes are the
    chunk's capacity class and nothing else, which
    ``tests/unit/test_chunk_row_avals.py`` pins: that is why warming one chunk
    per class covers every chunk of that class.

    A K-class table (``tables.n_classes > 1``) needs ``n_fine_rot``: its rows
    carry projection ids (:func:`_row_projection_ids`) and the class layout.
    """

    image_capacity = int(chunk.image_capacity)
    host_chunk = materialize_chunk(tables, chunk)
    segment_offsets_np = _chunk_segment_offsets(tables, chunk, n_fine_trans)
    image_row_start_np = segment_offsets_np.astype(np.int64)[:image_capacity] // int(n_fine_trans)
    image_row_count_np = (
        segment_offsets_np.astype(np.int64)[1:] - segment_offsets_np.astype(np.int64)[:-1]
    ) // int(n_fine_trans)
    n_classes = int(tables.n_classes)
    if n_classes > 1 and n_fine_rot is None:
        raise ValueError("a K-class chunk needs n_fine_rot for its projection ids")
    classes = None
    if n_classes > 1:
        classes = _chunk_class_layout(
            host_chunk, chunk, n_classes=n_classes, n_fine_trans=n_fine_trans, place=place
        )
    mstep = None
    if int(tables.n_slots) > 1:
        mstep = _chunk_mstep_layout(host_chunk, place=place)
    return _ChunkRowArrays(
        row_image_local=place.array(host_chunk["row_image_local"], jnp.int32),
        row_fine_rot=place.array(
            _row_projection_ids(host_chunk, n_fine_rot if n_classes > 1 else None), jnp.int32
        ),
        row_log_prior=place.array(host_chunk["row_log_prior"], jnp.float32),
        row_mask_bits=place.array(host_chunk["row_mask_bits"], jnp.uint32),
        row_mask_mode=place.array(host_chunk["row_mask_mode"], jnp.int8),
        image_ids=place.array(host_chunk["image_ids"], jnp.int32),
        n_valid_rows=place.scalar(host_chunk["n_valid_rows"], jnp.int32),
        n_valid_images=place.scalar(host_chunk["n_valid_images"], jnp.int32),
        segment_offsets=place.array(segment_offsets_np, jnp.int32),
        image_row_start=place.array(image_row_start_np, jnp.int64),
        image_row_count=place.array(image_row_count_np, jnp.int64),
        classes=classes,
        mstep=mstep,
    )


def _make_chunk_translation_sqdist(
    default,
    *,
    translation_prior_centers_np,
    image_indices,
    image_capacity,
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
    image_capacity = int(image_capacity)
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


def _make_chunk_stage_operands(recon, translation_sqdist_ang) -> _ChunkStageOperands:
    """Name one chunk's operands out of whatever produced them.

    ``recon`` is the per-chunk preparation's dict, the once-per-half gather's
    dict, or -- for the compile-ahead warm-up -- the same gather's output under
    ``jax.eval_shape``, which is a dict of the same keys holding avals.
    """

    return _ChunkStageOperands(
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
    )


def chunk_program_path() -> str:
    """Which programs a chunk of this half will actually submit.

    ``"fused"`` is the opt-in single chunk program, ``"per-stage"`` the default
    path's three glue programs, ``"eager"`` the loose dispatch with no program
    to warm. The compile-ahead warm-up reads this so it cannot warm a path the
    loop does not take: warming the fused program while the loop runs the
    per-stage one is not an error, it simply buys nothing, and it did exactly
    that until 2026-09-20.
    """

    if _chunk_jit_enabled():
        return "fused"
    if _resident_glue_jit_enabled():
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
    time and buys nothing; that happened once, on the fused-versus-per-stage
    split, and was found by reading a census rather than by a test.
    """

    if path == "fused":
        return (_run_resident_chunk_program,)
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
def _make_mstep_block_inputs(rows, posterior, *, n_slots: int) -> "_MstepBlockInputs":
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
    of blocks (:func:`_slot_mstep_blocks`) and the rows of a boundary block
    outside it get no weight. Grouping the live rows into fewer blocks changes
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
    return _MstepBlockInputs(
        row_image_local=rows.row_image_local[order],
        kernel_row_image_ids=posterior.kernel_row_image_ids[order],
        row_posterior=posterior.row_posterior[order],
        row_fine_rot=rows.row_fine_rot[order],
        projections=None,
        slot_offsets=slot_offsets,
    )


def _slot_mstep_blocks(blocks: "_MstepBlockInputs", slot_index: int, *, spec):
    """Accumulator slot ``slot_index``'s M-step rows: ``(blocks, first block, block count)``.

    A slot owns the live rows ``[lo, hi)`` of the M-step order
    (:func:`_make_mstep_block_inputs`); its blocks are the ones that overlap
    them, and :func:`_resident_mstep_block_at` gives the other rows of a
    boundary block no weight. Under ``static_block_trip`` a single slot runs
    the whole capacity instead. Block counts are device scalars, so the program
    stays keyed on the capacity class.
    """

    block_rows = int(spec.mstep_block_rows)
    offsets = blocks.slot_offsets
    if int(spec.n_slots) == 1 and spec.static_block_trip:
        return (
            blocks._replace(class_row_range=offsets[0:2]),
            jnp.int32(0),
            jnp.int32(int(spec.row_capacity) // block_rows),
        )
    lo, hi = offsets[slot_index], offsets[slot_index + 1]
    first = jax.lax.div(lo, jnp.int32(block_rows))
    stop = jax.lax.div(hi + jnp.int32(block_rows - 1), jnp.int32(block_rows))
    n_blocks = jnp.where(hi > lo, stop - first, jnp.int32(0))
    return blocks._replace(class_row_range=offsets[slot_index : slot_index + 2]), first, n_blocks


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
) -> _ChunkProgramSpec:
    """The static key of one chunk program.

    The last four environment-read fields are the reason this is a function
    and not a literal at each call site: a warm-up that read them at a
    different moment, or not at all, would key its program differently from the
    loop's and warm nothing.
    """

    return _ChunkProgramSpec(
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
        kernel_ctf_probs=_kernel_ctf_probs_enabled(),
        wavg_power_per_image=_wavg_power_per_image_enabled(),
        block_unroll=_chunk_block_unroll(),
        static_block_trip=_chunk_static_block_trip_enabled(),
        reuse_coarse_normalization=bool(reuse_coarse_normalization),
        firstiter_cc=bool(firstiter_cc),
        n_slots=int(n_slots),
        mstep_subtract_ctf_projection=bool(mstep_subtract_ctf_projection),
        n_classes=int(n_classes),
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
    """Queue one chunk program per capacity class the plan will run.

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

    from relax.sparse_pass2.resident_operands import ResidentHalfOperands

    # Warm the programs the configured path will actually submit. The fused
    # chunk program is opt-in and off by default; the per-stage path with the
    # glue JIT on is what production runs, and it is three programs. Warming the
    # wrong one is not an error, it is simply useless, so the path is decided
    # here, before any work, and named in the log line.
    path = chunk_program_path()
    use_chunk_jit = path == "fused"
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
            if use_chunk_jit:
                return _checked(
                    [
                        (
                            _run_resident_chunk_program,
                            (_row, operand_avals, table_avals, carry_avals),
                            {"spec": _spec},
                        )
                    ]
                )
            # The per-stage path is the default, and it runs three programs.
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
) -> _ChunkStageTables:
    """Assemble the iteration-global tables every chunk of a half reads.

    One builder, called by the chunk loop with the real arrays and by the
    compile-ahead warm-up with their shape/dtype stand-ins. Assembling the
    tuple twice would be a place for the warm-up to drift from the loop: a
    warmed program with one field's dtype wrong is never used, which costs
    compile time and is invisible unless someone reads the hit rate.
    """

    return _ChunkStageTables(
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

    image_capacity = int(chunk.image_capacity)
    n_valid_images = int(chunk.n_valid_images)
    image_indices = np.asarray(image_indices)

    batch_data, ctf_params, fetched_indices = fetch_indexed_batch(
        experiment_dataset, image_indices
    )
    order = _reorder_permutation(fetched_indices, image_indices, image_capacity)
    padded_fetched_indices = _pad_batch_to_capacity(np.asarray(fetched_indices), image_capacity)
    prepared = _prepare_bucket_io(
        experiment_dataset,
        jnp.asarray(_pad_batch_to_capacity(batch_data, image_capacity)),
        _pad_batch_to_capacity(ctf_params, image_capacity),
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
        np.arange(image_capacity) < n_valid_images, dtype=bool
    )
    score_shifted_cc = cc_half_batch_norm = None
    if normalized_cc:
        # The compact engine's operands (sparse_pass2_bucketed.py, the
        # normalized_cc branch): the translated corrected score tile in the
        # score window, and the -0.5 * |image|^2 evidence offset.
        tile = shifted_corrected_score_half.reshape(image_capacity, int(n_fine_trans), -1)
        if not windowed_prepare:
            tile = tile[:, :, jnp.asarray(score_window_indices, dtype=jnp.int32)]
        score_shifted_cc = _zero_padded_images(tile[permutation], valid_images)
        cc_half_batch_norm = _zero_padded_images(
            (0.5 * jnp.reshape(batch_norm, (image_capacity,)).real)[permutation], valid_images
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
        ),
        permutation,
        valid_images,
        exact_positions_device,
        image_capacity=int(image_capacity),
        n_fine_trans=int(n_fine_trans),
    )
    if int(shifted_recon.shape[-1]) != int(n_recon_windowed):
        raise ValueError(
            "reconstruction tile pixel count does not match the reconstruction window: "
            f"{int(shifted_recon.shape[-1])} vs {int(n_recon_windowed)}"
        )

    # Padded slots keep scale 1 so the Wavg kernel never divides by zero; their
    # posterior is zero, so the value is never observable.
    scale_chunk = np.ones(image_capacity, dtype=np.float32)
    if scale_corrections_np is not None:
        scale_chunk[:n_valid_images] = np.asarray(
            scale_corrections_np[image_indices], dtype=np.float32
        )
    group_ids_chunk = np.full(image_capacity, -1, dtype=np.int32)
    if group_ids_np is not None:
        group_ids_chunk[:n_valid_images] = np.asarray(
            group_ids_np[image_indices], dtype=np.int32
        )
    optics_groups_chunk = None
    if optics_groups_np is not None:
        optics_groups_chunk = np.zeros(image_capacity, dtype=np.int32)
        optics_groups_chunk[:n_valid_images] = np.asarray(
            optics_groups_np[image_indices], dtype=np.int32
        )

    translation_prior = jnp.asarray(
        np.zeros((image_capacity, int(n_fine_trans)), dtype=np.float32)
        if fine_translation_prior_2d is None
        else _pad_batch_to_capacity(
            np.asarray(fine_translation_prior_2d)[image_indices], image_capacity
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
    }


def _chunk_timing_enabled() -> bool:
    """Whether to log per-chunk occupancy, block count and synchronised wall."""

    return parse_env_flag(_CHUNK_TIMING_ENV, default=False)


def _chunk_jit_enabled() -> bool:
    """Whether the chunk body runs as one jitted program (T14).

    Opt-in: ``RELAX_SPARSE_PASS2_RESIDENT_CHUNK_JIT=1`` selects it, and the
    per-stage path is both the default and the oracle. Both paths call the same
    stage helpers on the same operands, so only the JIT boundary and the M-step
    loop's trip mechanism differ.

    Measured at 0bedf7672 on one H100 (jobs 14147789 hp3, 14147877 early,
    14147878 end-to-end). The program removes the chunk body's eager dispatch
    (267 -> 2 per chunk at hp3, 740 -> 2 at early) and 77% of its XLA glue
    launches, and the warm chunk loop is 0.9% (hp3) to 4.2% (early) faster. It
    is off by default because the end-to-end gate does not hold: with a cold
    persistent cache the fused program compiles once per (capacity class, pixel
    class) inside the chunk loop, and over the 16 iterations the 10k run took
    with an identical trajectory that cost 22.5 s (+1.9%), 84% of it in the two
    iterations that introduced a new pixel class. Turn it on with a warm
    persistent cache, or after that first-use compile is cheaper.
    """

    return parse_env_flag(_CHUNK_JIT_ENV, default=False)


def _chunk_block_unroll() -> int:
    """M-step blocks emitted per device-loop iteration; 1 keeps today's loop.

    XLA:GPU executes every ``while`` iteration by copying the loop predicate to
    the host and synchronizing, so the device loop drains the pipeline once per
    iteration: the hp3 nsys arm of job 14147789 charges 264 such
    ``cuStreamSynchronize`` calls, 5.94 s, to one warm half, and the early arm
    charges 1.88 ms to each of 1867 blocks. An unroll of ``u`` divides that by
    ``u`` and multiplies the live pixel-axis transients by ``u``, so it is a
    memory trade, not a free one; emitting every block cost 54.5 GiB.
    """

    raw = os.environ.get(_CHUNK_BLOCK_UNROLL_ENV, "").strip()
    if not raw:
        return 1
    value = int(raw)
    if value < 1:
        raise ValueError(f"{_CHUNK_BLOCK_UNROLL_ENV} must be >= 1, got {value}")
    return value


def _kernel_ctf_probs_enabled() -> bool:
    """Whether the M-step's ``ctf_probs`` comes from the kernel's fourth output."""

    return parse_env_flag(_KERNEL_CTF_PROBS_ENV, default=False)


def _wavg_power_per_image_enabled() -> bool:
    """Whether the Wavg rectangle is squared per image rather than per row.

    Default **on** since the P4-F/P4-G merge. P4-G measured the two forms
    bitwise on GPU at production shapes, 0 ULP over 6.3 million values, and the
    hoisted form 2.2 s per steady hp3 iteration faster;
    ``RELAX_SPARSE_PASS2_RESIDENT_WAVG_POWER_PER_IMAGE=0`` restores the
    per-row square as the oracle that equality is measured against.
    """

    return parse_env_flag(_WAVG_POWER_PER_IMAGE_ENV, default=True)


def _resident_operands_verify_enabled() -> bool:
    """Whether to check the first chunk of a half against the per-chunk path."""

    return parse_env_flag(_RESIDENT_OPERANDS_VERIFY_ENV, default=False)


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
    cuda_backproject,
) -> None:
    """Prove one real chunk's resident operands equal the per-chunk preparation.

    The two paths differ in *when* the translation is applied, so the check
    translates the resident per-image operands with the primitives
    ``_prepare_bucket_io`` uses and compares against its own pre-shifted tiles.
    A mismatch raises: this runs only in a diagnostic arm, and a silent
    difference here would be a changed reconstruction operand.
    """

    image_capacity = int(np.asarray(reference_recon["shifted_recon"]).shape[0])
    angles = jnp.asarray(translation_angles, dtype=jnp.float32)
    indices = jnp.asarray(recon_pixel_indices, dtype=jnp.int32)
    if bpref_recon_operand:
        translated_recon = cuda_backproject.relion_translate_bpref_f32(
            jnp.asarray(resident_recon["recon_image"], dtype=jnp.complex64),
            jnp.asarray(resident_recon["recon_weight"], dtype=jnp.float32),
            angles,
            indices,
            image_shape,
        )
    else:
        translated_recon = cuda_backproject.relion_translate_score_f32(
            jnp.asarray(resident_recon["recon_image"], dtype=jnp.complex64),
            angles,
            indices,
            image_shape,
        )
    translated_noise = cuda_backproject.relion_translate_score_f32(
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


def _resident_operands_requested() -> bool:
    """Whether the per-image operands are prepared once per half (T16).

    Default on. ``RELAX_SPARSE_PASS2_RESIDENT_OPERANDS=0`` keeps the per-chunk
    ``_prepare_bucket_io`` preparation and the XLA tile reduction, which are the
    oracle for every bitwise comparison of the new path.
    """

    return parse_env_flag(_RESIDENT_OPERANDS_ENV, default=True)


def _resident_glue_jit_enabled() -> bool:
    """Whether the chunk loop's three stages are dispatched as jitted programs.

    Default on. With the flag off every stage runs as the loose sequence of
    eager operations the per-stage path used before P3-A: the same functions,
    the same order, only without the enclosing ``jax.jit``. That form is the
    oracle for the bitwise comparison, so it is kept rather than deleted.
    """

    return parse_env_flag(_RESIDENT_GLUE_JIT_ENV, default=True)


def _carry_aval_probe_enabled() -> bool:
    """Whether to check the static carry avals against a shape probe."""

    return parse_env_flag(_CARRY_AVAL_PROBE_ENV, default=False)


_DEVICE_INT32_CACHE: dict[int, jax.Array] = {}


def _scalar_operand(value, dtype) -> jax.Array:
    """Put a host scalar on the device without an eager conversion.

    ``jnp.asarray(np.int32(7), dtype=jnp.int32)`` dispatches a
    ``convert_element_type`` because a NumPy scalar is not an array; the same
    value wrapped in a 0-d NumPy array of the target dtype is transferred with
    no primitive at all. The chunk driver builds two of these per chunk, so on
    the early state that was 834 eager dispatches over two iterations for two
    integers whose value never leaves the host.
    """

    return jnp.asarray(np.asarray(value, dtype=jnp.dtype(dtype)))


def _device_int32(value: int) -> jax.Array:
    """A device int32 scalar, made once per distinct value for the process.

    The M-step block program takes its row offset as a device operand so that
    one program serves every block of a capacity class. Building that scalar
    with ``jnp.asarray`` inside the loop would put one eager dispatch back per
    block, which is the cost this program exists to remove; the offsets are a
    handful of multiples of the block size, so they are made once and reused.
    Single-device only, which is what the EM engines run on; a multi-device
    process falls through to a fresh array.
    """

    key = int(value)
    if len(jax.devices()) != 1:
        return jnp.asarray(key, dtype=jnp.int32)
    cached = _DEVICE_INT32_CACHE.get(key)
    if cached is None:
        cached = jnp.asarray(key, dtype=jnp.int32)
        _DEVICE_INT32_CACHE[key] = cached
    return cached


def _chunk_static_block_trip_enabled() -> bool:
    """Whether the chunk program's M-step loop runs the whole row capacity.

    Default off. The live-block bound skips blocks whose rows are all chunk
    padding; at the hp3 state that is about a quarter of the pixel-axis work
    (row occupancy 0.73 over the two measured iterations). Both forms are one
    program per capacity class, and a padded block contributes exact zeros, so
    the two are bitwise equal; the flag exists to measure that claim.
    """

    return parse_env_flag(_CHUNK_STATIC_BLOCKS_ENV, default=False)


# ---------------------------------------------------------------------------
# One program per chunk (T14)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ChunkProgramSpec:
    """Everything the chunk program is keyed on.

    Only capacity classes, pixel counts and resolved configuration appear here,
    so two chunks of one class share a program whatever their occupancy: the
    valid row and image counts travel as device scalars.
    """

    row_capacity: int
    image_capacity: int
    n_fine_trans: int
    n_score_pixels: int
    n_recon_pixels: int
    n_rect: int
    mstep_block_rows: int
    adaptive_fraction: float
    current_size: int
    mstep_current_size: int
    image_shape: tuple
    # The M-step adjoint's max_r: mstep_current_size // 2 on one grid, a ReferenceSphereClip
    # for images on another grid (relax.helpers.adjoint).
    mstep_max_r: object
    recon_volume_shape: tuple
    max_adjoint_block_bytes: int
    stats_config: object
    use_rfloat_ctf_wavg: bool
    # T16: whether the M-step's weighted sums come from the translate-and-sum
    # kernel on unshifted operands, and whether its reconstruction operand takes
    # the BPref convention (a weight) or the score convention (no weight).
    use_translate_sum_kernel: bool
    bpref_recon_operand: bool
    # Whether ``ctf_probs`` comes from the kernel's fourth output instead of the
    # XLA statement the per-chunk path used. Off by default: the kernel's own
    # translation mass is bitwise against ``jnp.sum`` only at some shapes.
    kernel_ctf_probs: bool
    # Whether the Wavg rectangle's image power is squared once per image and
    # gathered, instead of gathered and squared once per row. Bitwise either
    # way; see ``_WAVG_POWER_PER_IMAGE_ENV``.
    wavg_power_per_image: bool
    # M-step blocks emitted per device-loop iteration.
    block_unroll: int
    # Trip mechanism of the M-step block loop. False (default) bounds the loop
    # by the chunk's live block count, a device scalar, so the padded blocks the
    # per-stage loop breaks out of are skipped; True runs the full capacity as
    # the ticket's literal form does. Both trace one program per capacity class.
    static_block_trip: bool
    # Zero oversampling: the posterior keeps every weight, divides by the
    # retained coarse sum and reports the coarse winner and Pmax
    # (_CoarseNormalizationReuse). The tables carry the arrays.
    reuse_coarse_normalization: bool = False
    # RELION --firstiter_cc: normalized-CC scores and a winner-take-all
    # posterior (ml_optimiser.cpp:8844-8858, :9266-9293).
    firstiter_cc: bool = False
    n_slots: int = 1
    # VDAM (--grad): BPref takes the residual shift(img) - ctf * proj, RELION's
    # cuda_kernel_backproject3D_SGD (BP.cuh:406-560); see _resident_block_residual.
    mstep_subtract_ctf_projection: bool = False
    # RELION Class3D classes; rows carry the class axis when there is more than one.
    n_classes: int = 1


class _MstepOnlyStatsConfig(NamedTuple):
    """The single statistics field the M-step block body reads.

    ``run_resident_mstep_blocks`` runs the M-step alone for a caller that owns
    its own scoring, posterior and statistics (local search, T12). Handing it
    the shell count in the shape ``_ChunkProgramSpec`` expects keeps one M-step
    body without making that caller build a full statistics configuration.
    """

    n_shells: int
    n_optics_groups: int = 1


class _ChunkRowArrays(NamedTuple):
    """One chunk's row-aligned tables and its runtime extents."""

    row_image_local: jax.Array  # int32 [C_R]
    # int32 [C_R]: the row's fine rotation; with K>1 classes its projection id
    # class * n_fine_rot + rotation; with streamed projections its cache slot.
    row_fine_rot: jax.Array
    row_log_prior: jax.Array  # float32 [C_R]
    row_mask_bits: jax.Array  # uint32 [C_R, n_mask_words], coarse-translation bitset per row
    row_mask_mode: jax.Array  # int8 [C_R]
    image_ids: jax.Array  # int32 [C_B], global image id, -1 when padded
    n_valid_rows: jax.Array  # int32 []
    n_valid_images: jax.Array  # int32 []
    segment_offsets: jax.Array  # int32 [C_B + 1], cell offsets
    image_row_start: jax.Array  # int64 [C_B], chunk-local first row of a slot
    image_row_count: jax.Array  # int64 [C_B], rows owned by a slot
    # K>1 only: the class axis of the chunk's rows.
    classes: "_ChunkClassLayout | None" = None
    # More than one accumulator slot only: the slot-major M-step order.
    mstep: "_ChunkMstepLayout | None" = None


class _ChunkClassLayout(NamedTuple):
    """The class axis of one chunk (K>1; docs/development/resident_segments.md).

    An image slot's rows are class-major, so each (slot, class) pair owns a
    contiguous sub-segment ``s = slot * K + class`` of the slot's posterior
    segment: the per-class evidence and winner are reductions over it. The
    M-step order is :class:`_ChunkMstepLayout`'s.
    """

    row_class: jax.Array  # int32 [C_R], 0 on padded rows
    row_segment: jax.Array  # int32 [C_R], slot * K + class; C_B * K on padded rows
    segment_offsets: jax.Array  # int32 [C_B * K + 1], cell offsets of each (slot, class)
    segment_row_start: jax.Array  # int64 [C_B * K], chunk-local first row of each (slot, class)


class _ChunkMstepLayout(NamedTuple):
    """The M-step visits a chunk's rows slot-major, so each accumulator slot's blocks write one BPref pair.

    Slot ``a = class + K * slot_offset[unit]`` (docs/development/resident_segments.md).
    """

    row_slot: jax.Array  # int32 [C_R], 0 on padded rows


class _ChunkStageOperands(NamedTuple):
    """One chunk's per-image operands, already padded to the image capacity."""

    score_input: jax.Array
    corr_img_score: jax.Array
    highres_xi2_half: jax.Array | None
    translation_prior: jax.Array
    # Exactly one reconstruction operand pair is populated. The per-chunk
    # preparation fills the pre-shifted ``[C_B, T, P]`` tiles; the once-per-half
    # preparation fills the unshifted ``[C_B, P]`` images T15's kernel takes,
    # with ``recon_weight`` set only in the exact-BPref configuration.
    shifted_recon: jax.Array | None
    shifted_noise: jax.Array | None
    recon_image: jax.Array | None
    recon_weight: jax.Array | None
    noise_image: jax.Array | None
    ctf2_over_nv_recon: jax.Array
    direct_ctf_rfloat_recon: jax.Array | None
    image_power_shells: jax.Array | None  # float64 [C_B, n_shells]
    relion_norm_high_shell: jax.Array
    raw_translated_wavg_rectangle: jax.Array
    raw_translated_wavg_for_atomic: jax.Array
    scale: jax.Array
    group_ids: jax.Array
    translation_sqdist_ang: jax.Array | None
    # int32 [C_B] optics-group row of each image, only with a [G, P] noise table.
    optics_groups: jax.Array | None = None
    # RELION --firstiter_cc only: the translated corrected score tile
    # [C_B, T, P_score] and 0.5 * |image|^2 per image (the evidence offset).
    score_shifted_cc: jax.Array | None = None
    cc_half_batch_norm: jax.Array | None = None


class _CoarseNormalizationReuse(NamedTuple):
    """The coarse pass's float32 normalization, winner and Pmax, per image of the half.

    With ``--adaptive_oversampling 0`` RELION's fine pass keeps the coarse
    ``sum_weight`` and max (acc_ml_optimiser_impl.h:2868), keeps every weight
    (``significant_weight = sorted[0]``, :3590) and reports Pmax as the coarse
    ``max_weight / sum_weight`` (:3268-3269, :4223). See docs/math/zero_oversampling.md.
    """

    sum_weight: jax.Array  # float64 [n_images]
    max_posterior: jax.Array  # [n_images]
    winner_cell: jax.Array  # int64 [image capacity], segment-relative r_local * T + t


class _ChunkStageTables(NamedTuple):
    """Iteration-global device tables every chunk of a half reads."""

    projection_score_cache: jax.Array
    projection_recon_cache: jax.Array
    projection_recon_abs2_cache: jax.Array
    mstep_grid: jax.Array
    coarse_parent_grid: jax.Array
    fine_translation_parent: jax.Array
    half_weights: jax.Array
    translation_angles: jax.Array
    full_to_compact: jax.Array
    noise_variance_for_noise: jax.Array
    shell_indices_noise: jax.Array
    exact_positions: jax.Array
    recon_pixel_indices: jax.Array
    relion_x_half_recon_indices: jax.Array
    shell_indices_half: jax.Array
    wavg_shell_indices: jax.Array
    wavg_scale_pixel_mask: jax.Array
    # Streamed projections: the global fine rotation id of each chunk-local
    # cache slot (rows then carry slots, not ids). None when the caches are the
    # per-iteration fine-grid caches, where the slot is the id.
    cache_slot_fine_rot: jax.Array | None = None
    # Zero oversampling only: the retained coarse normalization (None otherwise).
    coarse_reuse: _CoarseNormalizationReuse | None = None


class _ChunkPosterior(NamedTuple):
    """Stages 1-4 of one chunk."""

    row_posterior: jax.Array  # float32 [C_R, T]
    min_diff2: jax.Array  # real [C_B]
    class_log_z: jax.Array  # float64 [C_B]
    best_log_score: jax.Array  # float32 [C_B]
    best_cell_index: jax.Array  # int64 [C_B]
    max_posterior: jax.Array  # real [C_B]
    kernel_row_image_ids: jax.Array  # int32 [C_R], -1 on padded rows
    row_is_valid: jax.Array  # bool [C_R]
    # K>1 only: the (slot, class) sub-segment reductions.
    classes: "_ChunkClassPosterior | None" = None


class _ChunkClassPosterior(NamedTuple):
    """Each (image slot, class) sub-segment's log-Z and own winner, flat ``slot * K + class``."""

    log_z: jax.Array  # float64 [C_B * K], -inf for a class without candidates
    best_log_score: jax.Array  # float32 [C_B * K]
    best_cell_index: jax.Array  # int64 [C_B * K], sub-segment-relative r_local * T + t


class _ChunkMstepCarry(NamedTuple):
    """Loop carry of the M-step block loop."""

    Ft_y: jax.Array
    Ft_ctf: jax.Array
    wavg_triplet_pixels: jax.Array  # float32 [C_B, P_rect, 3]
    noise_shells: jax.Array  # float64 [n_shells], [G, n_shells] with G optics groups
    a2_per_image: jax.Array  # real [C_B]
    xa_per_image: jax.Array  # real [C_B]
    # K>1 only: each image's scale sums, every class masked by its own
    # data_vs_prior_class > 3 (:func:`_fold_class_scale_sums`).
    scale_xa_per_image: jax.Array | None = None  # float64 [C_B]
    scale_aa_per_image: jax.Array | None = None  # float64 [C_B]


@jax.jit
def _fold_class_scale_sums(mstep: "_ChunkMstepCarry", class_mask_rect) -> "_ChunkMstepCarry":
    """Move one class's Wavg XA/AA into the per-image scale sums under its own mask.

    RELION keeps XA and AA per class and adds them to the particle's scale sums
    only where that class's ``data_vs_prior_class > 3``
    (acc_ml_optimiser_impl.h:4893-4912); the diff2 channel is summed over
    classes. The class's blocks have just accumulated its XA/AA pixels, so they
    are masked and summed here and the two channels cleared for the next class.
    """

    triplet = mstep.wavg_triplet_pixels
    mask = jnp.asarray(class_mask_rect, dtype=bool).reshape(1, -1)
    zero = jnp.float32(0.0)
    xa = jnp.sum(jnp.where(mask, triplet[:, :, 0], zero).astype(jnp.float64), axis=1)
    aa = jnp.sum(jnp.where(mask, triplet[:, :, 1], zero).astype(jnp.float64), axis=1)
    return mstep._replace(
        wavg_triplet_pixels=triplet.at[:, :, :2].set(zero),
        scale_xa_per_image=mstep.scale_xa_per_image + xa,
        scale_aa_per_image=mstep.scale_aa_per_image + aa,
    )


def _resident_chunk_posterior(
    rows: _ChunkRowArrays,
    operands: _ChunkStageOperands,
    tables: _ChunkStageTables,
    *,
    spec: _ChunkProgramSpec,
    cuda_backproject,
) -> _ChunkPosterior:
    """Gather, cached projection, score and the segmented RELION posterior.

    Identical calls to the per-stage path; factored out so the jitted program
    and its oracle cannot drift apart.
    """

    row_capacity = int(spec.row_capacity)
    image_capacity = int(spec.image_capacity)
    n_fine_trans = int(spec.n_fine_trans)

    row_index = jnp.arange(row_capacity, dtype=jnp.int32)
    row_is_valid = row_index < rows.n_valid_rows
    kernel_row_image_ids = jnp.where(row_is_valid, rows.row_image_local, jnp.int32(-1))
    image_index = jnp.arange(image_capacity, dtype=jnp.int32)
    chunk_image_ids = jnp.where(image_index < rows.n_valid_images, image_index, jnp.int32(-1))

    if spec.firstiter_cc:
        return _resident_chunk_posterior_firstiter_cc(
            rows,
            operands,
            tables,
            spec=spec,
            cuda_backproject=cuda_backproject,
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
        logical_current_size=jnp.asarray(spec.current_size, dtype=jnp.int32),
        row_capacity=row_capacity,
        image_capacity=image_capacity,
        n_fine_trans=n_fine_trans,
        n_score_pixels=int(spec.n_score_pixels),
    )
    scores_flat = jnp.asarray(scored.scores, dtype=jnp.float32).reshape(-1)

    reuse = bool(spec.reuse_coarse_normalization)
    if reuse:
        # rows.image_ids are the chunk's global image ids, -1 in padded slots,
        # whose values the segmented kernels never read.
        image_slot = jnp.maximum(rows.image_ids, jnp.int32(0))
        coarse_sum_weight = tables.coarse_reuse.sum_weight[image_slot]
        external_sum_weight = coarse_sum_weight.astype(jnp.float32)
    else:
        external_sum_weight = jnp.ones((image_capacity,), dtype=jnp.float32)
    log_z = cuda_backproject.sparse_pass2_segmented_log_z_f64(
        scores_flat, rows.segment_offsets, rows.n_valid_images
    )
    posterior = cuda_backproject.sparse_pass2_segmented_posterior_f32(
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
        # (sparse_pass2_bucketed.py, reuse_coarse_normalization): RELION keeps the
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
            scores_flat.reshape(row_capacity, n_fine_trans),
            rows,
            row_is_valid,
            n_segments=image_capacity * int(spec.n_classes),
            n_classes=int(spec.n_classes),
            cuda_backproject=cuda_backproject,
        )
    return _ChunkPosterior(
        row_posterior=jnp.asarray(reconstruction_probs, dtype=jnp.float32).reshape(
            row_capacity, n_fine_trans
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


def _class_sub_segment_posterior(
    scores, rows, row_is_valid, *, n_segments: int, n_classes: int, cuda_backproject
):
    """Each (slot, class) sub-segment's log-Z and first maximum, for the per-class statistics.

    The scores are the joint-min-centred ones the image's posterior normalizes,
    so a class's log-Z plus the image's offset is its absolute evidence and
    ``best - logZ_image`` its share of Pmax. The log-Z is the segmented kernel
    the image posterior uses, run on the sub-segments; the winner is the first
    maximum in row order, as for the image (:func:`_winner_take_all_cells`).
    """

    layout = rows.classes
    log_z = cuda_backproject.sparse_pass2_segmented_log_z_f64(
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
    return _ChunkClassPosterior(
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
    rows: _ChunkRowArrays,
    operands: _ChunkStageOperands,
    tables: _ChunkStageTables,
    *,
    spec: _ChunkProgramSpec,
    cuda_backproject,
    row_is_valid,
    kernel_row_image_ids,
) -> _ChunkPosterior:
    """RELION's ``--firstiter_cc`` iteration: normalized-CC scores, winner takes all.

    RELION zeroes every weight but the best one (ml_optimiser.cpp:9266-9293;
    acc_ml_optimiser_impl.h:2868-2960), so the posterior is one-hot at each
    image's highest CC and Pmax is 1. The winner is the first maximum in the
    image's segment order, which is the compact engine's ``jnp.argmax`` over its
    bucket rows in the same order (``_winner_take_all_bucket_probs``). log-Z is
    the log-sum-exp of the CC scores, the compact engine's reported evidence.
    """

    from relax.sparse_pass2.resident_scoring import score_resident_chunk_normalized_cc

    row_capacity = int(spec.row_capacity)
    image_capacity = int(spec.image_capacity)
    n_fine_trans = int(spec.n_fine_trans)
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
        row_capacity=row_capacity,
        n_fine_trans=n_fine_trans,
        block_rows=int(spec.mstep_block_rows),
    )
    scores = jnp.asarray(scored.scores, dtype=jnp.float32)
    scores_flat = scores.reshape(-1)
    log_z = cuda_backproject.sparse_pass2_segmented_log_z_f64(
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
    return _ChunkPosterior(
        row_posterior=winner,
        min_diff2=scored.min_diff2,
        class_log_z=jnp.asarray(log_z, dtype=jnp.float64),
        best_log_score=best_log_score,
        best_cell_index=best_cell_index,
        max_posterior=has_winner.astype(jnp.float32),
        kernel_row_image_ids=kernel_row_image_ids,
        row_is_valid=row_is_valid,
    )


def _cached_block_projections(tables: _ChunkStageTables, block_fine_rot):
    """The global pass's block projections: three gathers by fine-rotation id.

    Split out so the M-step block body takes its projections as operands. The
    global pass keeps gathering them out of the per-iteration caches exactly
    where it did before; local search (T12) has no cacheable fine grid and
    slices the projections it computed for the chunk's own rows instead.
    """

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
    operands: _ChunkStageOperands,
    tables: _ChunkStageTables,
    carry: _ChunkMstepCarry,
    spec: _ChunkProgramSpec,
    cuda_backproject,
) -> _ChunkMstepCarry:
    """One pixel-axis row block: weighted sums, Wavg, noise, both adjoints.

    Statement for statement the body the per-stage loop ran inline; every path
    calls this one copy, so the only differences between them are how the
    block's rows are sliced (static Python slice versus ``dynamic_slice``) and
    where its projections come from. ``block_projections`` is the block's
    ``(projection, |projection|^2, M-step rotations)``: the global pass passes
    ``_cached_block_projections(tables, block_fine_rot)``, local search passes
    a slice of the projections it computed for this chunk.
    """

    proj, proj_abs2, block_mstep_rotations = block_projections
    logical_recon_pixels = jnp.asarray(spec.n_recon_pixels, dtype=jnp.int32)
    logical_rect_pixels = jnp.asarray(spec.n_rect, dtype=jnp.int32)
    block_optics_groups = (
        None if operands.optics_groups is None else operands.optics_groups[block_row_image]
    )

    if spec.use_translate_sum_kernel:
        summed, summed_masked, ctf_probs, _probs_sum_t = _resident_block_weighted_sums_kernel(
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
            cuda_backproject=cuda_backproject,
        )
    else:
        summed, summed_masked, ctf_probs, _probs_sum_t = _resident_block_weighted_sums(
            block_posterior,
            block_row_image,
            operands.shifted_recon,
            operands.shifted_noise,
            operands.ctf2_over_nv_recon,
        )

    if spec.mstep_subtract_ctf_projection:
        summed = _resident_block_residual(
            summed, _probs_sum_t, proj, operands.ctf2_over_nv_recon, block_row_image
        )

    # RELION Wavg triplet in the flat-row layout, then its rotation atomics.
    # The host tail picks the sequential RELION reducer when the pass carries
    # the RFLOAT CTF operand and the algebraic form otherwise; follow the same
    # branch, resolved on the host into the program's static configuration.
    if not spec.use_rfloat_ctf_wavg:
        exact_terms = _resident_block_wavg_algebraic_terms(
            proj,
            proj_abs2,
            summed_masked,
            ctf_probs,
            tables.noise_variance_for_noise,
            operands.scale,
            operands.raw_translated_wavg_for_atomic,
            block_posterior,
            block_row_image,
            block_optics_groups,
        )
    else:
        exact_terms = cuda_backproject.relion_wavg_sequential_runtime_flat_rows_triplet_f32(
            jnp.asarray(proj, dtype=jnp.complex64),
            block_kernel_ids,
            jnp.asarray(operands.direct_ctf_rfloat_recon, dtype=jnp.float32),
            jnp.asarray(operands.scale, dtype=jnp.float32),
            jnp.asarray(operands.raw_translated_wavg_for_atomic, dtype=jnp.complex64),
            block_posterior,
            logical_recon_pixels,
        )
    rectangle_terms = _resident_block_wavg_rectangle_terms(
        exact_terms,
        operands.raw_translated_wavg_rectangle,
        block_posterior,
        block_row_image,
        tables.exact_positions,
        power_per_image=bool(spec.wavg_power_per_image),
    )
    wavg_triplet_pixels = (
        cuda_backproject.relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32(
            rectangle_terms,
            block_kernel_ids,
            carry.wavg_triplet_pixels,
            logical_rect_pixels,
        )
    )

    block_shells, block_a2, block_xa = _resident_block_noise_and_norm(
        proj,
        proj_abs2,
        summed_masked,
        ctf_probs,
        tables.noise_variance_for_noise,
        tables.shell_indices_noise,
        block_row_image,
        block_optics_groups,
        n_shells=int(spec.stats_config.n_shells),
        image_capacity=int(spec.image_capacity),
    )

    Ft_y = _accumulate_adjoint_block_chunked(
        summed,
        block_mstep_rotations,
        carry.Ft_y,
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
        log_label="resident-y-window",
    )
    Ft_ctf = _accumulate_adjoint_block_chunked(
        ctf_probs,
        block_mstep_rotations,
        carry.Ft_ctf,
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
        log_label="resident-ctf-window",
    )
    return carry._replace(
        Ft_y=Ft_y,
        Ft_ctf=Ft_ctf,
        wavg_triplet_pixels=wavg_triplet_pixels,
        noise_shells=carry.noise_shells + block_shells,
        a2_per_image=carry.a2_per_image + block_a2,
        xa_per_image=carry.xa_per_image + block_xa,
    )


def _mstep_block_operand_dtypes(
    operands: _ChunkStageOperands,
    tables: _ChunkStageTables,
    *,
    spec: _ChunkProgramSpec,
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
            tables.projection_recon_cache.dtype,
            tables.projection_recon_abs2_cache.dtype,
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
    tables: _ChunkStageTables,
    *,
    spec: _ChunkProgramSpec,
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
    image_capacity = int(spec.image_capacity)

    def probe(summed_masked, ctf_probs, proj, proj_abs2, noise, shells, row_image, row_groups):
        return _resident_block_noise_and_norm(
            proj,
            proj_abs2,
            summed_masked,
            ctf_probs,
            noise,
            shells,
            row_image,
            row_groups,
            n_shells=int(spec.stats_config.n_shells),
            image_capacity=image_capacity,
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
    tables: _ChunkStageTables,
    *,
    spec: _ChunkProgramSpec,
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
    operands: _ChunkStageOperands,
    tables: _ChunkStageTables,
    *,
    spec: _ChunkProgramSpec,
    projection_dtypes=None,
) -> _ChunkMstepCarry:
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

    image_capacity = int(spec.image_capacity)
    dtypes = _mstep_block_operand_dtypes(
        operands, tables, spec=spec, projection_dtypes=projection_dtypes
    )
    if _carry_aval_probe_enabled():
        _check_mstep_carry_avals(tables, spec=spec, dtypes=dtypes)
    wavg_triplet_pixels, noise_shells, a2_per_image, xa_per_image = _zero_block_partials(
        (
            ((image_capacity, int(spec.n_rect), 3), jnp.dtype(jnp.float32)),
            (_noise_shell_shape(spec.stats_config), jnp.dtype(dtypes["noise_shells"])),
            ((image_capacity,), jnp.dtype(dtypes["a2"])),
            ((image_capacity,), jnp.dtype(dtypes["xa"])),
        )
    )()
    class_scale = {}
    if int(getattr(spec, "n_classes", 1)) > 1:
        class_scale = dict(
            scale_xa_per_image=jnp.zeros((image_capacity,), dtype=jnp.float64),
            scale_aa_per_image=jnp.zeros((image_capacity,), dtype=jnp.float64),
        )
    return _ChunkMstepCarry(
        Ft_y=Ft_y,
        Ft_ctf=Ft_ctf,
        wavg_triplet_pixels=wavg_triplet_pixels,
        noise_shells=noise_shells,
        a2_per_image=a2_per_image,
        xa_per_image=xa_per_image,
        **class_scale,
    )


class _MstepBlockInputs(NamedTuple):
    """Chunk-wide row arrays the M-step block program slices its block out of.

    ``row_fine_rot`` and ``projections`` are alternatives, and exactly one is
    populated: the global pass hands the fine-rotation ids and the program
    gathers the block's projections out of the per-iteration caches, while
    local search has no cacheable fine grid and hands the block's projections
    directly. ``None`` is a pytree structure, so the two callers key different
    programs without a flag.
    """

    row_image_local: jax.Array  # int32 [C_R]
    kernel_row_image_ids: jax.Array  # int32 [C_R]
    row_posterior: jax.Array  # float32 [C_R, T]
    row_fine_rot: jax.Array | None  # int32 [C_R]
    projections: tuple | None  # (proj, |proj|^2, M-step rotations) of one block
    # The rows [start, stop) of the accumulator slot these blocks accumulate;
    # the rows of a boundary block outside it get no weight. None where the
    # caller hands the block its rows directly (local search).
    class_row_range: jax.Array | None = None  # int32 [2]
    # Each slot's live rows in this order (:func:`_make_mstep_block_inputs`).
    slot_offsets: jax.Array | None = None  # int32 [n_slots + 1]


def _resident_mstep_block_at(
    block_start,
    blocks: _MstepBlockInputs,
    operands: _ChunkStageOperands,
    tables: _ChunkStageTables,
    carry: _ChunkMstepCarry,
    *,
    spec: _ChunkProgramSpec,
    cuda_backproject,
) -> _ChunkMstepCarry:
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

    if blocks.projections is None:
        block_projections = _cached_block_projections(tables, take(blocks.row_fine_rot))
    else:
        block_projections = blocks.projections
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
        operands=operands,
        tables=tables,
        carry=carry,
        spec=spec,
        cuda_backproject=cuda_backproject,
    )


@partial(jax.jit, static_argnames=("spec",), donate_argnums=(4,))
def _resident_mstep_block_program(
    block_start,
    blocks: _MstepBlockInputs,
    operands: _ChunkStageOperands,
    tables: _ChunkStageTables,
    carry: _ChunkMstepCarry,
    *,
    spec: _ChunkProgramSpec,
) -> _ChunkMstepCarry:
    """One M-step block as one program, keyed on the capacity and pixel class.

    Same statements, same order, same dtypes as the loose dispatch; the only
    change is where the JIT boundary sits. The carry is donated so the two
    half-volumes the windowed adjoint accumulates into keep being updated in
    place, as they are when the adjoint FFI is dispatched on its own.
    """

    from relax.cuda import kernels as em_cuda_kernels

    return _resident_mstep_block_at(
        block_start,
        blocks,
        operands,
        tables,
        carry,
        spec=spec,
        cuda_backproject=em_cuda_kernels,
    )


@partial(jax.jit, static_argnames=("spec",))
def _resident_chunk_posterior_program(
    rows: _ChunkRowArrays,
    operands: _ChunkStageOperands,
    tables: _ChunkStageTables,
    *,
    spec: _ChunkProgramSpec,
) -> _ChunkPosterior:
    """:func:`_resident_chunk_posterior` as one program per capacity class.

    The stage's own arithmetic is unchanged; what leaves the chunk loop is the
    dozen eager operations around it -- the two ``arange`` row/image masks, the
    logical current size, the score reshape and the posterior's unit external
    weight -- each of which was a dispatch and a single-primitive program.
    """

    from relax.cuda import kernels as em_cuda_kernels

    return _resident_chunk_posterior(
        rows, operands, tables, spec=spec, cuda_backproject=em_cuda_kernels
    )


@partial(jax.jit, static_argnames=("spec",), donate_argnums=(0,))
def _resident_chunk_statistics_program(
    stats,
    rows: _ChunkRowArrays,
    operands: _ChunkStageOperands,
    tables: _ChunkStageTables,
    posterior: _ChunkPosterior,
    mstep: _ChunkMstepCarry,
    *,
    spec: _ChunkProgramSpec,
):
    """:func:`_resident_chunk_statistics` as one program per capacity class.

    The statistics accumulator is donated: it is a running total the driver
    rebinds every chunk, so updating it in place is what the loose dispatch
    already did through ``_accumulate_chunk_image_terms``.
    """

    return _resident_chunk_statistics(
        stats, rows, operands, tables, posterior, mstep, spec=spec
    )


def _resident_chunk_statistics(
    stats,
    rows: _ChunkRowArrays,
    operands: _ChunkStageOperands,
    tables: _ChunkStageTables,
    posterior: _ChunkPosterior,
    mstep: _ChunkMstepCarry,
    *,
    spec: _ChunkProgramSpec,
):
    """Fold one chunk's image-level terms and its padding sanity counter."""

    image_tables = _ChunkImageTables(
        shell_indices_half=tables.shell_indices_half,
        wavg_shell_indices=tables.wavg_shell_indices,
        wavg_scale_pixel_mask=tables.wavg_scale_pixel_mask,
        translation_sqdist_ang=operands.translation_sqdist_ang,
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

    chunk_operands = _ChunkImageOperands(
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
    stats = _accumulate_chunk_image_terms(
        stats, chunk_operands, image_tables, config=spec.stats_config
    )
    return stats._replace(
        invalid_best_rows=stats.invalid_best_rows + jnp.sum(invalid_best.astype(jnp.int64))
    )


@partial(jax.jit, static_argnames=("spec",), donate_argnums=(3,))
def _run_resident_chunk_program(
    rows: _ChunkRowArrays,
    operands: _ChunkStageOperands,
    tables: _ChunkStageTables,
    carry: tuple,
    *,
    spec: _ChunkProgramSpec,
):
    """Every device stage of one capacity chunk in one program.

    The Python chunk loop around this call contains only the host materialize,
    the host padding and the operand preparation: no ``block_until_ready``, no
    ``.item()``, no ``np.asarray`` of a device value. The M-step block loop is
    a ``lax.fori_loop`` whose trip count is the chunk's *live* block count, a
    device scalar, so the program is keyed on the capacity class alone while
    still skipping the padded blocks the per-stage loop breaks out of. A static
    trip count would instead run those blocks; at the hp3 state that is about
    half of the pixel-axis work, all of it multiplying a zero posterior.
    """

    from recovar import cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    Ft_y_total, Ft_ctf_total, stats = carry
    posterior = _resident_chunk_posterior(
        rows, operands, tables, spec=spec, cuda_backproject=em_cuda_kernels
    )

    block_rows = int(spec.mstep_block_rows)
    mstep = _initial_mstep_carry(Ft_y_total[0], Ft_ctf_total[0], operands, tables, spec=spec)
    blocks = _make_mstep_block_inputs(rows, posterior, n_slots=int(spec.n_slots))
    unroll = max(int(spec.block_unroll), 1)
    Ft_y_out, Ft_ctf_out = [], []
    for slot_index in range(int(spec.n_slots)):
        # Each slot's blocks accumulate into its own volumes; the per-image
        # Wavg, noise and norm partials carry on across slots.
        mstep = mstep._replace(Ft_y=Ft_y_total[slot_index], Ft_ctf=Ft_ctf_total[slot_index])
        class_blocks, first_block, n_blocks = _slot_mstep_blocks(blocks, slot_index, spec=spec)
        n_outer = jax.lax.div(n_blocks + jnp.int32(unroll - 1), jnp.int32(unroll))

        def outer(outer_index, carry_in, _blocks=class_blocks, _first=first_block):
            # ``unroll`` blocks per device-loop iteration: the predicate is read
            # back once per iteration, and a trailing block past the live count
            # carries only rows without weight (padding, or another class's).
            for offset in range(unroll):
                index = _first + outer_index * unroll + offset
                carry_in = _resident_mstep_block_at(
                    index * block_rows,
                    _blocks,
                    operands,
                    tables,
                    carry_in,
                    spec=spec,
                    cuda_backproject=cuda_backproject,
                )
            return carry_in

        mstep = jax.lax.fori_loop(0, n_outer, outer, mstep)
        if mstep.scale_xa_per_image is not None:
            # Slot ``class + K * group``: the scale sums are masked by the slot's class.
            mstep = _fold_class_scale_sums(mstep, tables.wavg_scale_pixel_mask[slot_index % int(spec.n_classes)])
        Ft_y_out.append(mstep.Ft_y)
        Ft_ctf_out.append(mstep.Ft_ctf)
    stats = _resident_chunk_statistics(
        stats, rows, operands, tables, posterior, mstep, spec=spec
    )
    return tuple(Ft_y_out), tuple(Ft_ctf_out), stats


# Divisors of the M-step block for classes with few live rows. The pixel-axis
# block costs in proportion to its rows whether or not they carry weight, and
# at 50k/256 one class of a chunk has about 40 live rows against a 2048-row
# block. Three sizes bound the programs a capacity class compiles.
_LIVE_BLOCK_DIVISORS = (64, 8, 1)
_MIN_LIVE_BLOCK_ROWS = 16


def _live_block_spec(spec: "_ChunkProgramSpec", n_live_rows: int) -> "_ChunkProgramSpec":
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
    rows: _ChunkRowArrays,
    operands: _ChunkStageOperands,
    tables: _ChunkStageTables,
    carry: tuple,
    *,
    spec: _ChunkProgramSpec,
    timing_hook=None,
):
    """Per-stage oracle: the same stages, dispatched one at a time.

    The host block loop reads each accumulator slot's live row range back from
    the device once per chunk (:func:`_make_mstep_block_inputs`), so it
    launches only the blocks that carry weight.

    Kept selectable by ``RELAX_SPARSE_PASS2_RESIDENT_CHUNK_JIT=0`` so the
    fused program can be compared against the path it replaces inside one
    process. ``timing_hook(name)`` is called after each stage when the chunk
    timing diagnostic is on; it synchronizes, so an arm that passes it is a
    diagnostic arm.
    """

    from relax.cuda import kernels as em_cuda_kernels

    glue_jit = _resident_glue_jit_enabled()
    Ft_y_total, Ft_ctf_total, stats = carry
    if glue_jit:
        posterior = _resident_chunk_posterior_program(rows, operands, tables, spec=spec)
    else:
        posterior = _resident_chunk_posterior(
            rows, operands, tables, spec=spec, cuda_backproject=em_cuda_kernels
        )
    if timing_hook is not None:
        timing_hook("posterior", posterior.row_posterior)

    mstep = _initial_mstep_carry(Ft_y_total[0], Ft_ctf_total[0], operands, tables, spec=spec)
    blocks = _make_mstep_block_inputs(rows, posterior, n_slots=int(spec.n_slots))
    slot_offsets = np.asarray(jax.device_get(blocks.slot_offsets), dtype=np.int64)
    Ft_y_out, Ft_ctf_out = [], []
    for slot_index in range(int(spec.n_slots)):
        row_lo, row_hi = int(slot_offsets[slot_index]), int(slot_offsets[slot_index + 1])
        mstep = mstep._replace(Ft_y=Ft_y_total[slot_index], Ft_ctf=Ft_ctf_total[slot_index])
        class_blocks = blocks._replace(class_row_range=blocks.slot_offsets[slot_index : slot_index + 2])
        block_spec = _live_block_spec(spec, row_hi - row_lo)
        block_rows = int(block_spec.mstep_block_rows)
        # Blocks past the slot's live rows hold only rows without weight, of
        # this slot or another: the weighted sums, the Wavg terms, the noise
        # partials and both adjoint scatters would add exact zeros.
        for start in range((row_lo // block_rows) * block_rows, row_hi, block_rows):
            if glue_jit:
                mstep = _resident_mstep_block_program(
                    _device_int32(start), class_blocks, operands, tables, mstep, spec=block_spec
                )
                continue
            mstep = _resident_mstep_block_at(
                _device_int32(start),
                class_blocks,
                operands,
                tables,
                mstep,
                spec=block_spec,
                cuda_backproject=em_cuda_kernels,
            )
        if mstep.scale_xa_per_image is not None:
            # Slot ``class + K * group``: the scale sums are masked by the slot's class.
            mstep = _fold_class_scale_sums(mstep, tables.wavg_scale_pixel_mask[slot_index % int(spec.n_classes)])
        Ft_y_out.append(mstep.Ft_y)
        Ft_ctf_out.append(mstep.Ft_ctf)
    if timing_hook is not None:
        timing_hook("mstep", (Ft_y_out, Ft_ctf_out))

    if glue_jit:
        stats = _resident_chunk_statistics_program(
            stats, rows, operands, tables, posterior, mstep, spec=spec
        )
    else:
        stats = _resident_chunk_statistics(
            stats, rows, operands, tables, posterior, mstep, spec=spec
        )
    return tuple(Ft_y_out), tuple(Ft_ctf_out), stats


def _run_resident_chunk(
    chunk,
    *,
    tables,
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
    mstep_max_r,
    recon_volume_shape,
    max_adjoint_block_bytes,
    noise_variance_for_noise,
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
    cuda_backproject,
    submitted_keys=None,
    optics_groups_np=None,
    relion_native_fine_units=False,
    coarse_reuse=None,
    firstiter_cc=False,
    mstep_subtract_ctf_projection=False,
):
    """Run every resident stage for one capacity chunk.

    ``submitted_keys``, when given, collects the ``(program name, spec)`` keys
    this chunk submits, so the driver can say how many of them the compile-ahead
    warm-up had already compiled. Recording costs a set insert per chunk.

    Returns the updated ``(Ft_y_total, Ft_ctf_total, stats)``. Host work inside
    is the chunk's materialize/pad, its operand preparation and the T7 offsets
    readback the segmented posterior performs internally; no per-chunk result
    is pulled.
    """

    row_capacity = int(chunk.row_capacity)
    image_capacity = int(chunk.image_capacity)
    n_valid_rows = int(chunk.n_valid_rows)
    n_valid_images = int(chunk.n_valid_images)
    image_indices = np.arange(chunk.image_start, chunk.image_stop, dtype=np.int64)
    timing = _chunk_timing_enabled()
    chunk_t0 = time.time()
    stage_t = {}
    if timing:
        jax.block_until_ready(Ft_y_total)
        chunk_t0 = time.time()

    n_classes = int(tables.n_classes)
    rows = _make_chunk_row_arrays(
        tables, chunk, n_fine_trans, place=_PLACE_ON_DEVICE, n_fine_rot=n_fine_rot
    )
    if stream_projection_fn is not None:
        (
            rows,
            cache_slot_fine_rot,
            (projection_score_cache, projection_recon_cache, projection_recon_abs2_cache),
            mstep_grid,
            coarse_parent_grid,
        ) = _stream_chunk_projections(
            rows,
            _row_projection_ids(materialize_chunk(tables, chunk), n_fine_rot if n_classes > 1 else None),
            n_valid_rows=n_valid_rows,
            row_capacity=row_capacity,
            project=stream_projection_fn,
            n_fine_rot=n_fine_rot,
            mstep_grid=mstep_grid,
            coarse_parent_grid=coarse_parent_grid,
        )
        if timing:
            jax.block_until_ready(projection_score_cache)
            stage_t["projections"] = time.time() - chunk_t0

    if resident_operands is None:
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
            current_size=current_size,
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
        image_slots = np.full(image_capacity, -1, dtype=np.int32)
        image_slots[:n_valid_images] = image_indices[:n_valid_images]
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
                    current_size=current_size,
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
                cuda_backproject=cuda_backproject,
            )
    if timing:
        jax.block_until_ready(
            recon["shifted_recon"] if resident_operands is None else recon["recon_image"]
        )
        stage_t["operands"] = time.time() - chunk_t0

    # The prior squared distances are a per-image host table; building them at
    # capacity here keeps the program keyed on the capacity class and takes the
    # eager device ops out of the chunk loop. Padded slots multiply a zero
    # posterior, so their value is never observable; they are zeroed anyway.
    translation_sqdist_ang = _make_chunk_translation_sqdist(
        image_tables.translation_sqdist_ang,
        translation_prior_centers_np=translation_prior_centers_np,
        image_indices=image_indices,
        image_capacity=image_capacity,
        n_valid_images=n_valid_images,
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
    )
    spec = _make_chunk_program_spec(
        row_capacity=row_capacity,
        image_capacity=image_capacity,
        n_fine_trans=n_fine_trans,
        n_score_pixels=n_score_pixels,
        n_recon_pixels=n_recon_windowed,
        n_rect=n_rect,
        mstep_block_rows=mstep_block_rows,
        adaptive_fraction=adaptive_fraction,
        current_size=current_size,
        mstep_current_size=mstep_current_size,
        mstep_max_r=mstep_max_r,
        image_shape=image_shape,
        recon_volume_shape=recon_volume_shape,
        max_adjoint_block_bytes=max_adjoint_block_bytes,
        stats_config=stats_config,
        use_rfloat_ctf_wavg=recon["direct_ctf_rfloat_recon"] is not None,
        use_translate_sum_kernel=resident_operands is not None,
        bpref_recon_operand=(
            resident_operands is not None and resident_operands.recon_weight is not None
        ),
        reuse_coarse_normalization=coarse_reuse is not None,
        firstiter_cc=firstiter_cc,
        n_slots=int(tables.n_slots),
        mstep_subtract_ctf_projection=bool(mstep_subtract_ctf_projection),
        n_classes=n_classes,
    )

    if submitted_keys is not None:
        submitted_keys.update(chunk_program_keys(chunk_program_path(), spec))

    use_jit = _chunk_jit_enabled()
    if use_jit:
        Ft_y_total, Ft_ctf_total, stats = _run_resident_chunk_program(
            rows, operands, stage_tables, (Ft_y_total, Ft_ctf_total, stats), spec=spec
        )
    else:
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
        )

    if timing:
        jax.block_until_ready((Ft_y_total, Ft_ctf_total, stats.wsum_sigma2_noise))
        total = time.time() - chunk_t0
        operands_s = stage_t.get("operands", 0.0)
        if use_jit:
            split = "program=%.3fs unroll=%d" % (total - operands_s, spec.block_unroll)
        else:
            posterior_end = stage_t.get("posterior", operands_s)
            mstep_end = stage_t.get("mstep", total)
            split = "score+posterior=%.3fs mstep=%.3fs statistics=%.3fs" % (
                posterior_end - operands_s,
                mstep_end - posterior_end,
                total - mstep_end,
            )
        blocks = sum(1 for s in range(0, row_capacity, int(mstep_block_rows)) if s < n_valid_rows)
        logger.info(
            "Resident pass-2 chunk timing: jit=%d images=%d/%d rows=%d/%d occupancy=%.3f "
            "mstep_blocks=%d/%d projections=%.3fs operands=%.3fs %s total=%.3fs",
            int(use_jit),
            n_valid_images, image_capacity, n_valid_rows, row_capacity,
            n_valid_rows / max(row_capacity, 1),
            blocks, row_capacity // int(mstep_block_rows),
            stage_t.get("projections", 0.0),
            operands_s,
            split,
            total,
        )
    return Ft_y_total, Ft_ctf_total, stats


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
    cuda_backproject,
    n_optics_groups: int = 1,
):
    """Walk one chunk's pixel axis in row blocks: Wavg, noise and both adjoints.

    Exactly one of ``block_projections`` and ``chunk_projections`` is given.

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

    if recon.get("shifted_recon") is None or recon.get("shifted_noise") is None:
        # Fail closed rather than hand ``None`` to the XLA weighted sums: a
        # ``recon`` without the pre-shifted tiles is T16's once-per-half
        # preparation, whose weighted sums are the translate-and-sum kernel's,
        # and this entry point has no kernel path (``spec`` below pins
        # ``use_translate_sum_kernel=False``).
        raise ValueError(
            "run_resident_mstep_blocks takes the per-chunk pre-shifted "
            "reconstruction operands ('shifted_recon'/'shifted_noise'); this "
            "chunk carries T16's once-per-half per-image operands instead "
            f"(keys present: {sorted(k for k, v in recon.items() if v is not None)}). "
            "Prepare the chunk with _prepare_chunk_reconstruction_operands, or "
            "give this entry point the kernel path before handing it resident "
            "operands."
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

    spec = _ChunkProgramSpec(
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
        # T12's local pass hands pre-shifted per-chunk operands, not T16's
        # once-per-half resident images, so the M-step body takes its weighted
        # sums from the XLA statement, as it did before T16.
        use_translate_sum_kernel=False,
        bpref_recon_operand=False,
        kernel_ctf_probs=False,
        wavg_power_per_image=_wavg_power_per_image_enabled(),
        block_unroll=1,
        static_block_trip=False,
    )
    operands = _ChunkStageOperands(
        score_input=None,
        corr_img_score=None,
        highres_xi2_half=None,
        translation_prior=None,
        shifted_recon=recon["shifted_recon"],
        shifted_noise=recon["shifted_noise"],
        # The unshifted per-image images are T16's once-per-half operands; the
        # local pass prepares pre-shifted tiles per chunk, so this pair stays
        # empty and the M-step body takes the XLA weighted sums.
        recon_image=None,
        recon_weight=None,
        noise_image=None,
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
    )
    tables = _ChunkStageTables(
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
        translation_angles=None,
        full_to_compact=None,
        noise_variance_for_noise=noise_variance_for_noise,
        shell_indices_noise=shell_indices_noise,
        exact_positions=exact_positions_device,
        # T16's kernel path addresses the reconstruction window by pixel index;
        # the XLA path this caller takes does not read it.
        recon_pixel_indices=None,
        relion_x_half_recon_indices=relion_x_half_recon_indices,
        shell_indices_half=None,
        wavg_shell_indices=None,
        wavg_scale_pixel_mask=None,
    )

    block_rows = int(mstep_block_rows)
    carry = None
    glue_jit = _resident_glue_jit_enabled()
    if chunk_projections is None:
        chunk_row_ids = None
        projection_dtypes = None
    else:
        # A host-side index vector, so it costs one transfer for the chunk and
        # no eager primitive: the block program slices it and gathers the rows.
        chunk_row_ids = jnp.asarray(np.arange(int(row_capacity), dtype=np.int32))
        projection_dtypes = (chunk_proj.dtype, chunk_proj_abs2.dtype)

    def start_carry(projections):
        return _initial_mstep_carry(
            Ft_y_total,
            Ft_ctf_total,
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
                _device_int32(start),
                _MstepBlockInputs(
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
            cuda_backproject=cuda_backproject,
        )
    if carry is None:
        # A chunk with no live rows: the accumulators are the incoming ones and
        # the partials are the zeros the statistics program expects.
        carry = start_carry(
            block_projections(0, block_rows) if chunk_projections is None else None
        )

    return (
        carry.Ft_y,
        carry.Ft_ctf,
        carry.wavg_triplet_pixels,
        carry.noise_shells,
        carry.a2_per_image,
        carry.xa_per_image,
    )
