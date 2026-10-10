"""Device-resident local-search pass 2 (T12).

The retired exact local engine (removed 2026-09-30) ran the order-4 local
iterations and the final all-data iteration of a K=1 auto-refine. The Phase 0
budget measured those at 313 s and 211 s of a 2113 s run with the GPU 93 % idle,
because the engine paid host time per bucket and recompiled per bucket shape.
This module runs the same pass on the
device-resident driver's stages: fixed-capacity flat-row chunks, one program
per capacity class, accumulators that stay on the device and one pull per half.

Scope
-----
:func:`compute_local_search_resident` is the **fine pass 2** of every K=1 local search
and its pass-1 parent probe; the fine pass is also the pass the final all-data iteration runs (``iteration_loop.py``
reaches it through the same ``local_outputs = _run_local_search_iteration`` call
site in ``relax/refinement/half_scoring.py``). Any other configuration raises
:class:`NotImplementedError` naming the missing piece: there is no other local
fine-pass engine to fall back to.

The pass-1 parent probe
-----------------------
``score_only`` with ``return_reconstruction_sample_indices`` runs RELION's local
pass 1 on the same stages (:func:`_run_resident_parent_probe`): score-only
operands, rows projected into the score window alone, the segmented float32
posterior with the adaptive fraction and RELION's ``maximum_significants`` cap
(:func:`_cap_significant_samples`), and per-image significant samples in the
exact engine's convention; no M-step and no accumulators.

Semantics that differ from the exact local engine, deliberately and measurably
------------------------------------------------------------------------------
The resident stages implement the *compact* K=1 pass-2 arithmetic, which is the
RELION-parity path for the global iterations. The exact local engine implements
a mathematically equivalent but differently factored arithmetic. Three
differences are real and are reported rather than assumed away:

1. **Scoring.** The local engine forms ``-0.5 * (cross + norms)`` with a JAX
   einsum over a pre-shifted ``(B, T, N)`` image tile and adds the image power
   ``-0.5 * batch_norm`` on the host afterwards. This driver calls RELION's
   fused translate-and-score CUDA kernel, which forms the whole
   ``sum |X_t - A|^2 * Minvsigma2`` in one pass and carries the out-of-window
   tail as the ``powerClass`` operand, then converts with RELION's common
   minimum. The posteriors are shift invariant, so the two agree up to float32
   association. The reported ``log_evidence`` and ``best_log_score`` are absolute:
   the common minimum is added back (pass 2's ``log_score_offset``, the probe's
   evidences).
2. **Significance.** The local engine sorts the float64 posterior on the host
   (``_find_significant_mask_full_sort``); this driver uses T7's segmented CUDA
   posterior, which is bitwise against the compact engine's rectangular
   handler. Cutoff ties can therefore be resolved differently.
3. **Noise shells.** The local engine accumulates the algebraic
   ``A2 - 2*XA`` residual and a separate image-power vector; this driver uses
   RELION's Wavg triplet with the direct low-shell residual, which is the
   compact engine's production form. The sum of the two vectors is the same
   quantity; the split between them, and its float32 rounding, are not.

Everything else - the candidate set and its order, RELION particle order for
the x-half BPref, the projector, the translation operand, the M-step
contraction and the scale/norm statistics - is the same code the accepted
paths run.
"""

from __future__ import annotations

import logging
import os
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from recovar.reconstruction import noise as noise_utils

from relax.helpers.adjoint import mstep_adjoint_max_r
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
from relax.helpers.preprocessing import half_translation_phase_table
from relax.helpers.projection import (
    relion_scale_correction_pixel_mask as _relion_scale_correction_pixel_mask,
)
from relax.helpers.scale_groups import prepare_scale_correction_groups
from relax.helpers.types import LocalEMResult, make_noise_stats, make_relion_stats
from relax.relion.optics_aberrations import dataset_magnification_is_anisotropic, dataset_needs_exact_ctf
from relax.relion.relion_projector_setup import (
    cast_relion_projector_for_execution,
    prepare_local_class_projector_slabs,
    prepare_local_projector_slab,
)
from relax.sparse_pass2 import local_search_records
from relax.sparse_pass2 import resident_pass2 as rp
from relax.sparse_pass2.resident_candidates import chunk_segment_offsets
from relax.sparse_pass2.resident_local_layout import (
    expand_local_chunk_mask_jnp,
    materialize_local_chunk,
    plan_local_capacity_chunks,
    tables_from_local_layout,
)
from relax.sparse_pass2.resident_operands import (
    ResidentOperandsUnsupported,
    require_unshifted_operand_support,
)
from relax.sparse_pass2.resident_scoring import (
    project_resident_live_rows,
    resident_row_projection_bytes,
    score_resident_chunk_in_row_blocks,
    score_resident_chunk_normalized_cc_in_row_blocks,
    score_resident_projected_chunk,
    score_resident_projected_chunk_normalized_cc,
)
from relax.sparse_pass2.resident_statistics import (
    finalize_statistics,
    make_resident_statistics,
    resolve_statistics_config,
)
from relax.sparse_pass2.sparse_pass2_bucket_io import (
    _prepare_bucket_io,
    _relion_cuda_score_translation_angles_if_available,
)
from relax.sparse_pass2.sparse_pass2_budget import (
    _max_adjoint_block_bytes_for_pass,
    _max_translation_tile_bytes_for_pass,
    _projection_cache_max_bytes_for_pass,
)
from relax.sparse_pass2.sparse_pass2_policy import (
    _RELION_WAVG_ATOMIC_SCALE_AA_ENV,
    ResidentConfigurationUnsupported,
    _relion_wavg_direct_modes,
)
from relax.sparse_pass2.sparse_pass2_projection_blocks import (
    _projection_kwargs_for_relion_score_window,
    projection_window_union,
)
from relax.sparse_pass2.sparse_pass2_scoring import (
    _relion_cuda_fine_full_to_compact_lookup,
    _relion_powerclass_noise_terms,
)
from relax.sparse_pass2.sparse_pass2_wavg import (
    _make_relion_wavg_rectangle,
)
from relax.sparse_pass2.sparse_pass2_window import (
    _pass2_half_weights,
    _pass2_window_setup,
    _sparse_pass2_window_setup,
)

logger = logging.getLogger(__name__)

_ROW_CAPACITY_LADDER_ENV = "RELAX_LOCAL_SEARCH_RESIDENT_ROW_CAPACITIES"
_IMAGE_CAPACITY_LADDER_ENV = "RELAX_LOCAL_SEARCH_RESIDENT_IMAGE_CAPACITIES"
# Diagnostic only: log one line per chunk with its occupancy, padding and
# per-stage seconds. It inserts ``block_until_ready`` between stages, so it
# serialises work that normally overlaps and inflates the loop; never use a
# profiled arm for a wall-time comparison.
_CHUNK_PROFILE_ENV = "RELAX_LOCAL_SEARCH_RESIDENT_CHUNK_PROFILE"
# P3-G, opt-in and off by default. Hand the M-step entry point the chunk's
# whole row arrays instead of a Python callback that slices a block out of them
# per block: the block program then takes its own ``dynamic_slice`` of the row
# ids and reads the rows inside the jit. The callback is kept as the oracle
# every bitwise comparison of this change is made against.
_BLOCK_ROW_PROGRAM_ENV = "RELAX_LOCAL_SEARCH_RESIDENT_BLOCK_ROW_PROGRAM"
_PROJECTION_CALL_MAX_BYTES_ENV = "RELAX_LOCAL_SEARCH_RESIDENT_PROJECTION_CALL_MAX_BYTES"
# One projector call's transient. The exact local engine budgets its own fused
# projection matmul at 4 GiB by default
# (local_batch_planning.EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB); use the same figure
# so the two engines reserve comparable headroom.
_DEFAULT_PROJECTION_CALL_MAX_BYTES = 4 * 1024**3


# The share of the chunk budget one projector call may take; the rest is the chunk's rows, tiles and M-step block.
_PROJECTION_CALL_BUDGET_SHARE = 0.5


def _projection_block_rows(projection_row_bytes: int, chunk_budget_bytes: int | None) -> int:
    """Rows per projector call: the per-call cap, lowered to ``_PROJECTION_CALL_BUDGET_SHARE`` of the chunk budget.

    The fixed 4 GiB call was most of the smallest chunk at a 16 GB card's full-box final pass (box 448: 4.28 GiB
    against a 3.63 GiB budget, relax#49). Fewer rows per call change only how the calls group the rows; each row's
    projection is the same. Large cards keep the cap.
    """

    max_bytes = _projection_call_transient_max_bytes()
    if chunk_budget_bytes is not None:
        max_bytes = min(max_bytes, int(_PROJECTION_CALL_BUDGET_SHARE * int(chunk_budget_bytes)))
    return max(1, int(max_bytes) // int(projection_row_bytes))


def _projection_call_transient_max_bytes() -> int:
    """Bytes one projector call may hold (:func:`relax.sparse_pass2.resident_pass2.projection_call_row_bytes`)."""

    raw = os.environ.get(_PROJECTION_CALL_MAX_BYTES_ENV, "").strip()
    if not raw:
        return _DEFAULT_PROJECTION_CALL_MAX_BYTES
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{_PROJECTION_CALL_MAX_BYTES_ENV} must be positive, got {raw!r}")
    return value

# Row capacities are powers of two so a chunk decomposes into whole M-step
# blocks; the ladder is truncated at run time by the projection byte budget,
# because the local route projects a chunk's rows rather than gathering a
# per-iteration cache.
_DEFAULT_ROW_CAPACITY_LADDER = (1024, 4096, 16384, 65536)
_DEFAULT_IMAGE_CAPACITY_LADDER = (32, 128, 512)

__all__ = [
    "compute_local_search_resident",
    "require_resident_local_configuration",
]


class _ClassProjector(NamedTuple):
    """One class's reference for a Class3D local pass: the volume, its slab (or its shape) and texture."""

    mean: object
    slab: object
    texture: object


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ResidentConfigurationUnsupported(
            "The device-resident local-search pass 2 does not implement this configuration: "
            f"{message}. There is no other local fine-pass engine."
        )


def require_resident_local_configuration(**kwargs) -> None:
    """Raise unless this is the K=1 local fine pass 2, or its pass-1 parent probe, in production shape."""

    if bool(kwargs["score_only"]):
        _require_parent_probe_configuration(**kwargs)
        return
    _require(
        not (bool(kwargs["disable_adjoint_y"]) or bool(kwargs["disable_adjoint_ctf"])),
        "a score-only pass has no M-step to make resident",
    )
    _require(bool(kwargs["mstep_relion_x_half"]), "the RELION x-half M-step is required")
    _require(bool(kwargs["accumulate_noise"]), "the production pass accumulates noise statistics")
    _require(
        kwargs["max_significants"] is None or int(kwargs["max_significants"]) <= 0,
        "a maximum_significants cap on the fine support is not in the segmented "
        "posterior's contract",
    )
    # Pass 2 of an oversampled search reconstructs from and accumulates the
    # pruned posterior; the zero-oversampling route keeps every weight for both
    # (RELION's symbolic second pass sets significant_weight to the minimum
    # weight: acc_ml_optimiser_impl.h:3590). One posterior serves both.
    _require(
        bool(kwargs["stats_use_reconstruction_probs"]) == bool(kwargs["reconstruct_significant_only"]),
        "the resident statistics stage accumulates the reconstruction posterior, so the "
        "statistics and the reconstruction must use the same (pruned or complete) weights",
    )
    _require(
        not bool(kwargs["use_float64_scoring"]) and not bool(kwargs["use_float64_projections"]),
        "float64 local search is a diagnostic mode",
    )
    _require(
        bool(kwargs["relion_exact_score_translation"]),
        "the fused translate-and-score kernel needs RELION translation angles",
    )
    _require(bool(kwargs["half_spectrum_scoring"]), "RELION half-spectrum scoring is required")
    _require(
        kwargs["relion_projector_half"] is not None
        and kwargs["relion_projector_r_max"] is not None,
        "the resident row projection uses the RELION PPref projector",
    )
    _require(
        kwargs["normalization_log_evidence"] is None,
        "an externally supplied normalizer belongs to the broad-denominator probe",
    )
    _require(
        not bool(kwargs["return_reconstruction_sample_indices"]),
        "significant-sample capture belongs to the pass-1 parent probe",
    )
    _require(
        kwargs["group_ids"] is not None,
        "the resident statistics stage accumulates RELION's group scale terms",
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
    # RELION scores its radial support at every current size, the final
    # all-data iteration's full box included (updateImageSizeAndResolutionPointers
    # cuts the FFTW rectangle's corners); both local engines take that window at
    # the box (window_at_box, aa03fd2), and the RELION Wavg rectangle requires it.
    _require(
        bool(kwargs["use_window"]),
        "the resident local pass scores RELION's radial window; a full-rectangle "
        "support is not RELION's scoring support and is not implemented here",
    )
    for name in rp._DIAGNOSTIC_DIR_ENVS:
        _require(
            not os.environ.get(name, "").strip(),
            f"the diagnostic dump {name} is set; this driver emits no dumps",
        )
    for name in rp._DIAGNOSTIC_FLAG_ENVS:
        _require(
            not parse_env_flag(name, default=False),
            f"the diagnostic flag {name} is set; this driver has no such arm",
        )


def _local_chunk_loop_pipelined() -> bool:
    """Whether the chunk loop enqueues chunk k+1 before chunk k's M-step (off while profiling).

    The chunk plan counts the second chunk's arrays exactly when it does
    (:func:`resident_pass2.resident_chunk_bytes`).
    """

    return not parse_env_flag(_CHUNK_PROFILE_ENV, default=False)


def _unshifted_operands_route(bucket_io_kwargs, *, window_indices, recon_window_indices) -> bool:
    """Whether this pass's chunks take the unshifted per-image operands (T16) or the translated tiles."""

    if not rp._resident_operands_requested():
        return False
    try:
        require_unshifted_operand_support(
            bucket_io_kwargs, window_indices=window_indices, recon_window_indices=recon_window_indices
        )
    except ResidentOperandsUnsupported as reason:
        logger.info("Resident local pass-2 keeps the pre-shifted translation tiles: %s", reason)
        return False
    return True


def _require_parent_probe_configuration(**kwargs) -> None:
    """Raise unless this is RELION's local pass 1 (the parent probe) in production shape."""

    _require(
        bool(kwargs["return_reconstruction_sample_indices"]),
        "a score-only local pass is the parent probe, which returns its significant samples",
    )
    # The probe keeps RELION's adaptive-fraction support; the segmented float32
    # posterior has no maximum_significants cap, and RELION's default (--maxsig
    # -1, ml_optimiser.cpp:1109) applies none. A cap stays on the exact engine.
    _require(
        kwargs["max_significants"] is None or int(kwargs["max_significants"]) <= 0,
        "a maximum_significants cap on the pass-1 support is not in the segmented "
        "posterior's contract",
    )
    _require(
        not bool(kwargs["use_float64_scoring"]) and not bool(kwargs["use_float64_projections"]),
        "float64 local search is a diagnostic mode",
    )
    _require(
        bool(kwargs["relion_exact_score_translation"]),
        "the fused translate-and-score kernel needs RELION translation angles",
    )
    _require(bool(kwargs["half_spectrum_scoring"]), "RELION half-spectrum scoring is required")
    _require(
        kwargs["relion_projector_half"] is not None
        and kwargs["relion_projector_r_max"] is not None,
        "the resident row projection uses the RELION PPref projector",
    )
    _require(
        kwargs["normalization_log_evidence"] is None,
        "an externally supplied normalizer belongs to the broad-denominator probe",
    )
    _require(bool(kwargs["use_window"]), "the probe scores RELION's current-size window")
    for name in rp._DIAGNOSTIC_DIR_ENVS:
        _require(
            not os.environ.get(name, "").strip(),
            f"the diagnostic dump {name} is set; this driver emits no dumps",
        )
    for name in rp._DIAGNOSTIC_FLAG_ENVS:
        _require(
            not parse_env_flag(name, default=False),
            f"the diagnostic flag {name} is set; this driver has no such arm",
        )


def _cap_row_capacity_ladder(
    ladder: tuple,
    *,
    n_score_pixels: int,
    n_recon_pixels: int,
    max_bytes: int,
) -> tuple:
    """Drop row classes whose resident projections exceed the projection budget.

    The global route amortizes one projection per fine rotation across every
    bucket that uses it; local search has no such grid, so a chunk's own
    projections are the memory that decides its row capacity. At the final
    all-data state (current size 256) a row costs about half a megabyte, which
    is why the largest classes disappear there and survive at current size 92.
    """

    per_row = resident_row_projection_bytes(
        n_score_pixels=n_score_pixels, n_recon_pixels=n_recon_pixels
    )
    cap = max(int(max_bytes) // max(per_row, 1), 1)
    kept = tuple(value for value in ladder if int(value) <= cap)
    return kept if kept else (int(ladder[0]),)


def compute_local_search_resident(
    data: local_search_records.LocalSearchData,
    local_layout,
    kernel: local_search_records.LocalSearchKernelPolicy,
    support: local_search_records.LocalSearchSupportPolicy,
    *,
    translation_prior_centers,
    symmetry_label: str,
) -> LocalEMResult:
    """Run one K=1 local-search fine pass 2 on the device-resident stages.

    ``data`` holds the half's images, reference and per-image corrections, ``local_layout`` the
    per-image candidate rows (:mod:`relax.local.local_layout`), ``kernel`` the window, projector and
    optics choices and ``support`` the posterior support and what is returned
    (:mod:`relax.sparse_pass2.local_search_records`). ``translation_prior_centers`` is each image's
    translation-prior centre, or None.

    ``data.noise_variance`` may be ``[G, P]`` rows of G optics groups with
    ``data.optics_group_ids`` giving each image's row, and ``kernel.reconstruction_volume_current_size``
    keeps the backprojector on the reference model size for images on another grid,
    as in the global resident pass.

    ``symmetry_label`` is the point group of the reconstruction. The local
    neighbourhoods in ``local_layout`` already include the symmetry mates of each
    prior direction; here the group only symmetrises the BPref accumulators.

    Returns the same :class:`~recovar.em.helpers.types.LocalEMResult` the exact
    local engine returns for this configuration. See the module docstring for
    the three arithmetic differences that are deliberate and for why the
    pass-1 parent probe is not routed here.

    ``kernel.nyquist_column_counting`` is the consistency option of the per-image sums
    (docs/math/relion_consistency_options.md): it changes the scoring weights and the image
    the noise statistics read, for the fine pass and its parent probe alike.

    ``kernel.wsum_current_size`` is the weighted sums' image size when the fine pass scores below it
    (``--strict_highres_exp``), as in :func:`resident_pass2._resident_pass2`; None is ``kernel.current_size``.

    ``kernel.firstiter_cc`` is RELION's ``--firstiter_cc`` iteration of a search that is local from its
    first iteration (``--sigma_ang``): both the parent probe and the fine pass score the
    normalized CC and keep only each image's best hidden variable (ml_optimiser.cpp:9266-9292),
    as the global ``--firstiter_cc`` pass does (resident_pass2 ``_resident_chunk_posterior_firstiter_cc``).
    In a Class3D the best hidden variable is over the classes of the layout too: RELION scores every
    class in its CC iteration when the references are given, and class 1 alone only when it generates
    seeds from one reference (``do_generate_seeds``, ml_optimiser.cpp:4392-4401), a start whose layout
    the caller has restricted to that class (``restrict_local_layout_classes``).
    """

    from recovar import cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels

    overall_t0 = time.time()
    image_shape = data.experiment_dataset.image_shape
    volume_shape = data.experiment_dataset.volume_shape
    # Anisotropic magnification: the M-step clips on RELION's rotated radius (adjoint.ReferenceSphereClip).
    anisotropic_magnification = dataset_magnification_is_anisotropic(data.experiment_dataset)

    # The resident drivers score RELION's window at every size, the box included
    # (window_at_box below), so the full box is an explicit current size here.
    current_size = (
        int(data.experiment_dataset.image_shape[0]) if kernel.current_size is None else kernel.current_size
    )
    wsum_current_size = current_size if kernel.wsum_current_size is None else int(kernel.wsum_current_size)
    score_mode = "normalized_cc" if kernel.firstiter_cc else "gaussian"
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
            reconstruction_current_size=kernel.reconstruction_current_size,
            half_spectrum_scoring=kernel.half_spectrum_scoring,
            square_window=kernel.square_window,
            relion_firstiter_score_mode=score_mode,
            use_exact_relion_gaussian=not kernel.firstiter_cc,
            use_float64_scoring=kernel.use_float64_scoring,
            # RELION's window at every size, including the box (a shape class reaches its
            # box before the reference does): the resident driver never scores a full half.
            window_at_box=True,
            reference_sphere_clip=kernel.reconstruction_image_radius is not None or anisotropic_magnification,
        )
    except NotImplementedError as exc:
        # Reported before any device work, like require_resident_local_configuration.
        raise ResidentConfigurationUnsupported(str(exc)) from exc

    # The retired exact local engine used this flag as passed rather than resolving it
    # against the environment, so do the same: it selects RELION's powerClass
    # shell spectrum for the image-power statistics and, with it, the
    # deterministic float64 norm reduction. It does not switch on RELION's
    # exact BPref operands here, because the exact local engine never enables
    # those on its production path.
    resolved_spectrum_norm = bool(kernel.source_faithful_spectrum_norm)
    scale_groups_available = data.group_ids is not None
    relion_wavg_atomic_scale_aa = bool(
        kernel.accumulate_noise
        and scale_groups_available
        and parse_env_flag(_RELION_WAVG_ATOMIC_SCALE_AA_ENV, default=True)
    )
    relion_wavg_atomic_direct_noise, relion_wavg_atomic_direct_norm = _relion_wavg_direct_modes(
        accumulate_noise=bool(kernel.accumulate_noise),
        scale_groups_available=scale_groups_available,
        scale_aa_enabled=bool(relion_wavg_atomic_scale_aa),
        direct_noise_only_default=True,
    )
    require_resident_local_configuration(
        score_only=support.score_only,
        disable_adjoint_y=support.disable_adjoint_y,
        disable_adjoint_ctf=support.disable_adjoint_ctf,
        mstep_relion_x_half=support.mstep_relion_x_half,
        accumulate_noise=kernel.accumulate_noise,
        reconstruct_significant_only=support.reconstruct_significant_only,
        max_significants=support.applied_max_significants,
        stats_use_reconstruction_probs=support.stats_use_reconstruction_probs,
        use_float64_scoring=kernel.use_float64_scoring,
        use_float64_projections=kernel.use_float64_projections,
        relion_exact_score_translation=kernel.relion_exact_score_translation,
        half_spectrum_scoring=kernel.half_spectrum_scoring,
        relion_projector_half=kernel.relion_projector_half,
        relion_projector_r_max=kernel.relion_projector_r_max,
        normalization_log_evidence=support.normalization_log_evidence,
        return_reconstruction_sample_indices=support.return_reconstruction_sample_indices,
        group_ids=data.group_ids,
        use_window=budget_window_spec.use_window,
        relion_wavg_atomic_scale_aa=relion_wavg_atomic_scale_aa,
        relion_wavg_atomic_direct_noise=relion_wavg_atomic_direct_noise,
        relion_wavg_atomic_direct_norm=relion_wavg_atomic_direct_norm,
    )
    _require(
        jax.default_backend() == "gpu" and cuda_backproject.custom_cuda_requested(),
        "every resident stage is a CUDA FFI target",
    )

    # ---- candidate rows ---------------------------------------------------
    # Read after the configuration checks, so a pass the driver refuses touches
    # nothing of the dataset beyond its shapes.
    n_images = int(data.experiment_dataset.n_units)
    table_t0 = time.time()
    tables = tables_from_local_layout(
        local_layout, rotation_dtype=precision_policy.score_real_dtype
    )
    if tables.n_images != n_images:
        raise ValueError(
            f"the local layout covers {tables.n_images} images but the half has {n_images}"
        )
    n_fine_trans = tables.n_trans
    fine_translations = np.asarray(
        tables.translation_grid, dtype=precision_policy.score_real_dtype
    )
    # Class3D local search (K>1): the layout's rows repeat per class; each class's rows are
    # projected with its own reference and backprojected into its own BPref.

    # ---- window, weights and lookups --------------------------------------
    window_setup = _sparse_pass2_window_setup(
        data.experiment_dataset,
        disc_type=kernel.disc_type,
        image_shape=image_shape,
        current_size=current_size,
        n_half=n_half,
        mstep_current_size=mstep_current_size,
        square_window=kernel.square_window,
        window_spec_kwargs=window_spec_kwargs,
        use_relion_x_half_mstep=True,
        log_label="Resident local pass-2",
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
        half_spectrum_scoring=kernel.half_spectrum_scoring,
        relion_firstiter_score_mode=score_mode,
        use_float64_scoring=kernel.use_float64_scoring,
        nyquist_column_counting=kernel.nyquist_column_counting,
    )
    del half_weights
    relion_score_full_to_compact = jnp.asarray(
        _relion_cuda_fine_full_to_compact_lookup(image_shape, current_size, window_indices_np),
        dtype=jnp.int32,
    )
    # The exact local engine casts sigma2 to the score real dtype before it
    # divides by it (local_big_jit.py, the non-BPref-operand branch). Passing a
    # float64 sigma2 through instead makes every downstream operand double:
    # ctf^2/sigma2 becomes float64, the translated reconstruction tile becomes
    # complex128, and _prepare_bucket_io then picks relion_translate_score_f64
    # where the exact engine picks the f32 kernel. That is a different
    # computation and twice the memory on the largest per-chunk array.
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(
        data.noise_variance, image_shape
    ).squeeze().astype(precision_policy.score_real_dtype)
    n_optics_groups = 1 if noise_variance_half.ndim == 1 else int(noise_variance_half.shape[0])
    optics_groups_np = None
    if n_optics_groups > 1:
        if data.optics_group_ids is None:
            raise ValueError("a per-optics-group noise table needs optics_group_ids")
        optics_groups_np = np.asarray(data.optics_group_ids, dtype=np.int32).reshape(-1)
        if optics_groups_np.shape != (n_images,) or np.any(optics_groups_np < 0) or np.any(
            optics_groups_np >= n_optics_groups
        ):
            raise ValueError(
                f"optics_group_ids must give each of {n_images} images a row of the "
                f"{n_optics_groups}-group noise table"
            )
    relion_score_translation_angles = _relion_cuda_score_translation_angles_if_available(
        fine_translations,
        image_shape,
        enabled=True,
        dtype=np.float32,
        angle_scale=kernel.relion_translation_angle_scale,
    )
    if relion_score_translation_angles is None:
        raise ValueError("the resident local scorer requires RELION translation angles")
    # Supply the phase table even in the windowed case. Handed ``None``, the
    # compact prepare builds it from a float64 cached lattice, which is a
    # complex128 table; the exact local engine builds it at the score real
    # dtype and windows it. Windowing a supplied table is what that engine
    # does, so supply it.
    translation_phases_half = half_translation_phase_table(
        fine_translations, image_shape, dtype=precision_policy.score_real_dtype
    ).astype(precision_policy.score_complex_dtype)

    n_shells = image_shape[0] // 2 + 1
    # With a weighted-sum size above the scoring size, the noise crop is that size's rectangle.
    noise_crop_indices = window_indices
    if wsum_current_size != current_size:
        noise_crop_indices = _make_relion_wavg_rectangle(
            image_shape, wsum_current_size, recon_window_indices, reconstruction_current_size=mstep_current_size
        ).centered_indices
    shell_indices_half = mask_relion_noise_shell_indices_to_current_window(
        make_relion_noise_shell_indices_half(image_shape),
        image_shape,
        wsum_current_size,
        noise_crop_indices,
    )
    shell_indices_noise = window_spec.recon_values(shell_indices_half)
    noise_variance_for_noise = window_spec.recon_values(noise_variance_half)
    # Class3D masks each class's scale sums by its own data_vs_prior_class
    # (acc_ml_optimiser_impl.h:4893-4912): ``[K, n_shells]``, or one curve for every class.
    scale_dvp_by_class = [data.scale_correction_data_vs_prior] * tables.n_classes
    if (
        tables.n_classes > 1
        and data.scale_correction_data_vs_prior is not None
        and np.ndim(data.scale_correction_data_vs_prior) == 2
    ):
        if int(np.shape(data.scale_correction_data_vs_prior)[0]) != tables.n_classes:
            raise ValueError(
                f"scale_correction_data_vs_prior must be one curve or ({tables.n_classes}, n_shells), "
                f"got {np.shape(data.scale_correction_data_vs_prior)}"
            )
        scale_dvp_by_class = [np.asarray(data.scale_correction_data_vs_prior)[k] for k in range(tables.n_classes)]
    scale_correction_pixel_mask = _relion_scale_correction_pixel_mask(
        scale_dvp_by_class[0],
        shell_indices_noise,
        n_shells=n_shells,
    )
    relion_wavg_rectangle = _make_relion_wavg_rectangle(
        image_shape,
        wsum_current_size,
        recon_window_indices,
        reconstruction_current_size=mstep_current_size,
    )
    n_rect = int(relion_wavg_rectangle.centered_indices.size)
    scale_pixel_mask_rect_np = np.zeros(n_rect, dtype=bool)
    scale_pixel_mask_rect_np[relion_wavg_rectangle.exact_positions] = np.asarray(
        scale_correction_pixel_mask, dtype=bool
    )
    class_scale_masks_rect = None
    if tables.n_classes > 1:
        class_masks_np = np.zeros((tables.n_classes, n_rect), dtype=bool)
        for k, dvp in enumerate(scale_dvp_by_class):
            class_masks_np[k, relion_wavg_rectangle.exact_positions] = np.asarray(
                _relion_scale_correction_pixel_mask(dvp, shell_indices_noise, n_shells=n_shells), dtype=bool
            )
        class_scale_masks_rect = jnp.asarray(class_masks_np)
    group_ids_np, n_scale_groups = prepare_scale_correction_groups(
        data.group_ids, data.scale_correction_group_count, n_images=n_images
    )

    # ---- accumulator layout (identical to the exact engine's x-half BPref) -
    volume_current_size = (
        mstep_current_size
        if kernel.reconstruction_volume_current_size is None
        else int(kernel.reconstruction_volume_current_size)
    )
    recon_volume_shape = relion_backprojector_volume_shape(
        volume_shape,
        kernel.reconstruction_padding_factor,
        current_size=volume_current_size,
    )
    recon_accum_shape = half_volume_accumulator_shape(recon_volume_shape)
    recon_volume_size = int(np.prod(recon_accum_shape))
    recon_y_accum_dtype, recon_ctf_accum_dtype = relion_x_half_mstep_accumulator_dtypes(
        data.experiment_dataset.dtype,
        use_relion_x_half_mstep=True,
    )
    # ---- projection setup -------------------------------------------------
    # The refinement loop hands local search a projector with a singleton class
    # axis; the exact local engine normalizes it with the same helper before
    # projecting, so do that here rather than letting the projector unpack a
    # 4-D shape.
    # The exact local engine selects the projection precision with
    # ``cast_relion_projector_for_execution`` (complex64 unless double
    # projection is requested) and then normalizes the slab. The precision
    # decides float32 projection arithmetic and whether the texture projector
    # or the vmapped fallback runs, so an arm with any other precision is both
    # a different computation and a differently timed one than its control.
    # Do exactly what the exact local engine does.
    relion_projector_half = cast_relion_projector_for_execution(
        kernel.relion_projector_half, use_float64_projections=kernel.use_float64_projections
    )
    class_slabs = None
    if tables.n_classes == 1:
        relion_projector_half = prepare_local_projector_slab(
            relion_projector_half, path_label="device-resident local projector path"
        )
    else:
        class_slabs = prepare_local_class_projector_slabs(
            relion_projector_half, tables.n_classes, path_label="device-resident local projector path"
        )
        relion_projector_half = class_slabs[0]
    logger.info(
        "Resident local pass-2 projector: slab dtype=%s shape=%s r_max=%s "
        "(the exact local engine's execution precision)",
        relion_projector_half.dtype,
        tuple(relion_projector_half.shape),
        kernel.relion_projector_r_max,
    )
    projection_kwargs = _projection_kwargs_for_relion_score_window(
        window_spec.projection_kwargs(return_abs2=False),
        use_relion_projector=True,
        current_size=current_size,
    )
    # The exact local engine passes all four of these to the same projector;
    # ``relion_texture_interp=None`` means "resolve as strict parity does",
    # which is RELION's CUDA texture interpolator.
    projection_kwargs["relion_texture_interp"] = kernel.projection_relion_texture_interp
    projection_kwargs["relion_acc_double_floorf_quirk"] = bool(
        kernel.projection_relion_acc_double_floorf_quirk
    )
    projection_kwargs["force_jax"] = False
    projection_kwargs["mask_current_image_disk"] = False
    projection_kwargs["relion_kernel"] = kernel.projection_relion_kernel
    # The pixels the projector computes for a chunk's rows: the two windows,
    # not the full half spectrum (10202 box 800: ~155k of 320,800 px per row).
    window_union = (
        projection_window_union(
            window_indices,
            recon_window_indices,
            image_shape=image_shape,
            projector_output_size=int(projection_kwargs["projector_output_size"]),
        )
        if projection_kwargs.get("projector_output_size") is not None
        else None
    )

    # One RELION projector texture for the whole pass, staged before the
    # capacity plan so the plan's device reading sees it. Projected per call,
    # the texture path stages the slab's two float32 CUDA arrays outside the
    # XLA pool on every chunk (7.9 GB at EMPIAR-10202 current size 626, which
    # failed with CUDA out of memory once the pool had grown, bigbox 14480549,
    # 14507538) and synchronizes the stream. RELION builds its projector
    # texture once per iteration.
    capacity_texture = _open_capacity_texture(
        relion_projector_half,
        relion_projector_r_max=kernel.relion_projector_r_max,
        projection_padding_factor=kernel.projection_padding_factor,
        projection_kwargs=projection_kwargs,
    )
    class_textures = [capacity_texture]
    try:
        if class_slabs is not None:
            class_textures += [
                _open_capacity_texture(
                    slab,
                    relion_projector_r_max=kernel.relion_projector_r_max,
                    projection_padding_factor=kernel.projection_padding_factor,
                    projection_kwargs=projection_kwargs,
                )
                for slab in class_slabs[1:]
            ]
        if capacity_texture is not None:
            # The staged texture serves every projection of the pass, which reads
            # only the slab's geometry from here on, so the device slab is released
            # (15.35 GiB at EMPIAR-10202's full box, where it left no free block for
            # the x-half accumulators: bigbox 14575557).
            relion_projector_half = jax.ShapeDtypeStruct(relion_projector_half.shape, relion_projector_half.dtype)
        class_projectors = None
        if class_slabs is not None:
            class_projectors = tuple(
                _ClassProjector(
                    mean=data.mean[k],
                    slab=slab if texture is None else jax.ShapeDtypeStruct(slab.shape, slab.dtype),
                    texture=texture,
                )
                for k, (slab, texture) in enumerate(zip(class_slabs, class_textures, strict=True))
            )
            del class_slabs
        # ---- per-image resident operands --------------------------------------
        bucket_io_kwargs = dict(
            noise_variance_half=noise_variance_half,
            fine_translations=fine_translations,
            config=config,
            n_trans=n_fine_trans,
            score_with_masked_images=True,
            half_spectrum_scoring=kernel.half_spectrum_scoring,
            image_corrections=data.image_corrections,
            scale_corrections=data.scale_corrections,
            image_pre_shifts=data.image_pre_shifts,
            use_float64_scoring=kernel.use_float64_scoring,
            score_only=False,
            score_mode=score_mode,
            window_indices=window_indices,
            recon_window_indices=recon_window_indices,
            translation_phases_half=translation_phases_half,
            relion_score_translation_angles=relion_score_translation_angles,
            return_windowed_shifted=windowed_prepare,
            # The --firstiter_cc iteration takes RELION's exact normalized-CC operands, as the
            # global K=1 CC pass does (k_class: relion_exact_fine_normalized_cc for one class).
            relion_exact_normalized_cc_operands=bool(kernel.firstiter_cc),
            # The exact local engine runs its production path with the plain
            # ``CTF^2 / sigma2`` operand order, not RELION's RFLOAT-square order,
            # so keep that here rather than silently switching operand families.
            # Datasets whose CTF needs the optics table (CTF-premultiplied images,
            # even Zernike terms, magnification) take the exact family, the only one
            # built from relax's exact RELION CTF rows, as the global pass 2 does.
            relion_exact_bpref_operands=dataset_needs_exact_ctf(data.experiment_dataset),
            noise_optics_groups=optics_groups_np,
        )
        if kernel.nyquist_column_counting != "relion":
            bucket_io_kwargs["nyquist_column_counting"] = kernel.nyquist_column_counting
        if support.score_only:
            # RELION's pass 1 (the local adaptive parent probe): score, posterior and
            # significance only, no M-step and no accumulators. See _run_resident_parent_probe.
            return _run_resident_parent_probe(
                tables=tables,
                experiment_dataset=data.experiment_dataset,
                bucket_io_kwargs=dict(bucket_io_kwargs, score_only=True),
                fine_translation_prior_2d=np.asarray(
                    tables.translation_log_prior, dtype=precision_policy.score_real_dtype
                ),
                half_weights=half_weights_windowed,
                full_to_compact=relion_score_full_to_compact,
                translation_angles=relion_score_translation_angles,
                n_score_pixels=int(n_windowed),
                mean=data.mean,
                volume_shape=volume_shape,
                disc_type=kernel.disc_type,
                projection_kwargs=projection_kwargs,
                projection_padding_factor=kernel.projection_padding_factor,
                relion_projector_half=relion_projector_half,
                relion_projector_r_max=kernel.relion_projector_r_max,
                relion_projector_capacity_texture=capacity_texture,
                class_projectors=class_projectors,
                precision_policy=precision_policy,
                n_fine_trans=n_fine_trans,
                adaptive_fraction=float(support.adaptive_fraction),
                windowed_prepare=windowed_prepare,
                window_indices=window_indices,
                image_shape=image_shape,
                current_size=current_size,
                source_faithful_spectrum_norm=resolved_spectrum_norm,
                return_profile=support.return_profile,
                overall_t0=overall_t0,
                max_significants=(
                    -1 if support.applied_max_significants is None else int(support.applied_max_significants)
                ),
                firstiter_cc=bool(kernel.firstiter_cc),
                window_union=(
                    projection_window_union(
                        window_indices,
                        image_shape=image_shape,
                        projector_output_size=int(projection_kwargs["projector_output_size"]),
                    )
                    if window_union is not None
                    else None
                ),
            )

        # Allocated after the slab is released and before the pass reads free device
        # memory for its chunk budget, which then counts them (up to 23 GiB at
        # EMPIAR-10202's full box).
        if tables.n_classes == 1:
            Ft_y_total = jnp.zeros(recon_volume_size, dtype=recon_y_accum_dtype)
            Ft_ctf_total = jnp.zeros(recon_volume_size, dtype=recon_ctf_accum_dtype)
        else:
            # One BPref per class (ml_optimiser.cpp, BPref[iclass]).
            Ft_y_total = [jnp.zeros(recon_volume_size, dtype=recon_y_accum_dtype) for _ in range(tables.n_classes)]
            Ft_ctf_total = [jnp.zeros(recon_volume_size, dtype=recon_ctf_accum_dtype) for _ in range(tables.n_classes)]

        # The operand family decides what a chunk holds, so it is chosen before the
        # plan; the per-chunk fallback in _start_resident_local_chunk stays as a guard.
        # The --firstiter_cc iteration scores the translated corrected score tile, which only
        # the pre-shifted family prepares (resident_pass2 refuses unshifted operands for it too).
        operand_route = {
            "unshifted": (not kernel.firstiter_cc) and _unshifted_operands_route(
                bucket_io_kwargs, window_indices=window_indices, recon_window_indices=recon_window_indices
            )
        }

        # ---- capacity plan ----------------------------------------------------
        row_ladder = _cap_row_capacity_ladder(
            parse_env_capacity_ladder(_ROW_CAPACITY_LADDER_ENV, _DEFAULT_ROW_CAPACITY_LADDER),
            n_score_pixels=n_windowed,
            n_recon_pixels=n_recon_windowed,
            max_bytes=_projection_cache_max_bytes_for_pass(device_memory_bytes),
        )
        # The accumulators already exist, so the budget reading counts them; the
        # plan records them so that its pass_bytes is the pass's total need.
        accumulator_bytes = tables.n_classes * rp.resident_accumulator_bytes(
            recon_volume_size, recon_y_accum_dtype, recon_ctf_accum_dtype
        )
        chunk_budget_bytes = rp.resident_chunk_budget_bytes()
        # One projector call holds the texture crop and the gathered rows
        # together (rp.projection_call_row_bytes); its block of rows is sized to
        # the per-call cap and its bytes enter the chunk plan.
        projector_output_size = projection_kwargs.get("projector_output_size")
        projector_slab_bytes = int(np.dtype(relion_projector_half.dtype).itemsize)
        projection_row_bytes = rp.projection_call_row_bytes(
            crop_pixels=(
                n_half
                if projector_output_size is None
                else int(projector_output_size) * (int(projector_output_size) // 2 + 1)
            ),
            output_pixels=n_half if window_union is None else int(window_union.indices.shape[0]),
            n_score_pixels=n_windowed,
            n_recon_pixels=n_recon_windowed,
            complex_bytes=projector_slab_bytes,
        )
        projection_block_rows = _projection_block_rows(projection_row_bytes, chunk_budget_bytes)
        tile_pixels = rp.chunk_translated_tile_pixels(
            unshifted_operands=operand_route["unshifted"],
            n_score_pixels=n_windowed,
            n_recon_pixels=n_recon_windowed,
            n_rect_pixels=n_rect,
            n_exact_rect_pixels=int(relion_wavg_rectangle.exact_positions.size),
            normalized_cc=bool(kernel.firstiter_cc),
            masked_scoring=True,
        )
        image_ladder = parse_env_capacity_ladder(_IMAGE_CAPACITY_LADDER_ENV, _DEFAULT_IMAGE_CAPACITY_LADDER)
        if not (operand_route["unshifted"] and chunk_budget_bytes is not None):
            # The fixed translation-tile budget bounds the pre-shifted route's
            # three recon tiles per image. An unshifted chunk holds only the
            # Wavg rectangle and its exact positions, which the joint plan below
            # counts against the measured budget; capped at three recon tiles,
            # the 10097 full-box final pass ran 32-image chunks, 2032 per half.
            image_ladder = rp.resident_image_capacity_start(
                image_ladder,
                n_fine_trans=n_fine_trans,
                n_recon_pixels=n_recon_windowed,
                max_tile_bytes=_max_translation_tile_bytes_for_pass(
                    device_memory_bytes, has_external_normalization=False
                ),
                chunk_budget_bytes=chunk_budget_bytes,
            )
        mstep_block_rows = rp._resolve_mstep_block_rows(
            n_recon_pixels=n_recon_windowed,
            max_block_bytes=_max_adjoint_block_bytes_for_pass(device_memory_bytes),
            row_capacity_ladder=row_ladder,
        )
        memory_plan = rp.plan_resident_chunk_memory(
            row_capacity_ladder=row_ladder,
            image_capacity_ladder=image_ladder,
            mstep_block_rows=mstep_block_rows,
            row_bytes=resident_row_projection_bytes(
                n_score_pixels=n_windowed, n_recon_pixels=n_recon_windowed
            ),
            n_fine_trans=n_fine_trans,
            n_recon_pixels=n_recon_windowed,
            budget_bytes=chunk_budget_bytes,
            pipelined=_local_chunk_loop_pipelined(),
            projection_transient_bytes=projection_block_rows * projection_row_bytes,
            fixed_bytes=accumulator_bytes,
            max_image_rows=rp.max_image_rows(tables.row_offsets),
            # An image past the largest row class runs alone in row blocks of that class
            # (_start_resident_local_chunk), as the global pass runs it (relax#49 follow-up).
            lone_row_bytes=rp.lone_chunk_row_bytes(int(n_fine_trans), class_rows=tables.n_classes > 1),
            **tile_pixels,
        )
        row_ladder = memory_plan.row_capacity_ladder
        image_ladder = memory_plan.image_capacity_ladder
        mstep_block_rows = memory_plan.mstep_block_rows
        n_projection_pixels = int(getattr(window_spec, "n_projection", n_recon_windowed))
        chunks = plan_local_capacity_chunks(
            tables,
            row_capacity_ladder=row_ladder,
            image_capacity_ladder=image_ladder,
        )
        table_s = time.time() - table_t0
        per_row_bytes = resident_row_projection_bytes(
            n_score_pixels=n_windowed, n_recon_pixels=n_recon_windowed
        )
        logger.info(
            "Resident local pass-2 plan: %d images, %d candidate rows, %d translations -> %d chunks "
            "(row capacities %s, image capacities %s, M-step block rows %d, projection block rows %d); "
            "row projections %.2f KiB/row, largest chunk %.2f GiB, projection window %d px, "
            "projector slab %d B/element; chunk peak %.2f GiB of a %s budget, pass need %.2f GiB "
            "(accumulators %.2f GiB); setup %.2fs",
            tables.n_images,
            tables.n_rows,
            n_fine_trans,
            len(chunks),
            ",".join(str(v) for v in row_ladder),
            ",".join(str(v) for v in image_ladder),
            mstep_block_rows,
            projection_block_rows,
            per_row_bytes / 1024.0,
            max((int(chunk.row_capacity) for chunk in chunks), default=0)
            * per_row_bytes
            / float(1024**3),
            n_projection_pixels,
            projector_slab_bytes,
            memory_plan.peak_bytes / float(1024**3),
            rp.format_budget_gib(memory_plan.budget_bytes),
            memory_plan.pass_bytes / float(1024**3),
            accumulator_bytes / float(1024**3),
            table_s,
        )

        fine_translation_prior_2d = np.asarray(
            tables.translation_log_prior, dtype=precision_policy.score_real_dtype
        )

        # ---- statistics accumulators ------------------------------------------
        # The rotation posterior is accumulated over the bins the rows use, not the
        # layout's whole rotation histogram: at MS2 box 512's final pass (I2,
        # HEALPix 9) that histogram is 9.60 GiB of float64 on the device, which
        # ran it out of memory (bench 14684161). It is expanded on the host.
        if tables.n_classes == 1:
            posterior_bin_ids, row_posterior_bin = np.unique(tables.row_posterior_id, return_inverse=True)
            row_posterior_bin = row_posterior_bin.astype(np.int32, copy=False)
        else:
            # The class axis needs the whole class-major histogram ([K, n_bins] at finalize).
            posterior_bin_ids = np.arange(tables.n_posterior_bins, dtype=np.int64)
            row_posterior_bin = np.asarray(tables.row_posterior_id, dtype=np.int32)
        stats_config = resolve_statistics_config(
            n_shells=n_shells,
            n_fine_trans=n_fine_trans,
            n_images=n_images,
            n_coarse_rot=_posterior_bin_capacity(posterior_bin_ids.size, n_classes=tables.n_classes),
            n_scale_groups=n_scale_groups,
            current_size=wsum_current_size,
            include_unweighted_high_shell=True,
            use_exact_relion_gaussian=True,
            relion_wavg_atomic_direct_noise=relion_wavg_atomic_direct_noise,
            relion_wavg_atomic_scale_aa=relion_wavg_atomic_scale_aa,
            accumulate_scale=scale_groups_available,
            source_faithful_spectrum_norm=resolved_spectrum_norm,
            n_optics_groups=n_optics_groups,
            n_classes=tables.n_classes,
            float32_bucketed_image_sums=False,
        )
        stats = make_resident_statistics(
            stats_config, max_posterior_dtype=precision_policy.score_real_dtype
        )
        image_tables = rp._ChunkImageTables(
            shell_indices_half=jnp.asarray(shell_indices_half, dtype=jnp.int32),
            wavg_shell_indices=jnp.asarray(relion_wavg_rectangle.shell_indices, dtype=jnp.int32),
            wavg_scale_pixel_mask=jnp.asarray(scale_pixel_mask_rect_np, dtype=bool),
            translation_sqdist_ang=None,
        )

        exact_positions_device = jnp.asarray(relion_wavg_rectangle.exact_positions, dtype=jnp.int32)
        rect_indices_device = jnp.asarray(relion_wavg_rectangle.centered_indices, dtype=jnp.int32)
        noise_variance_for_noise_device = jnp.asarray(noise_variance_for_noise)
        shell_indices_noise_device = jnp.asarray(shell_indices_noise, dtype=jnp.int32)
        max_adjoint_block_bytes = _max_adjoint_block_bytes_for_pass(device_memory_bytes)

        scale_corrections_np = (
            None
            if data.scale_corrections is None
            else np.asarray(data.scale_corrections, dtype=precision_policy.score_real_dtype)
        )
        translation_prior_centers_np = None
        if translation_prior_centers is not None:
            from relax.helpers.translation_prior import validate_translation_prior_centers

            translation_prior_centers_np = validate_translation_prior_centers(
                translation_prior_centers,
                n_images=n_images,
                n_dims=int(fine_translations.shape[1]),
            )

        # ---- chunk loop --------------------------------------------------------
        loop_t0 = time.time()
        # The chunks are software-pipelined: chunk k+1's operands, projections,
        # scores and posterior are enqueued before chunk k's M-step reads its live
        # row count back, so the device works through k+1's front while the host
        # waits on and dispatches k's M-step. The chunk-profile diagnostic finishes
        # every chunk at once so its stage timers stay per chunk.
        pipelined = _local_chunk_loop_pipelined()
        pending = None
        pending_alone = False
        n_lone_chunks = 0
        lone_block_sizes = set()
        for chunk in chunks:
            alone = rp.chunk_runs_alone(chunk, row_ladder)
            lone_block_rows = rp.lone_block_rows(chunk.row_capacity, row_ladder) if alone else None
            n_lone_chunks += int(alone)
            if alone:
                lone_block_sizes.add(lone_block_rows)
            if pending is not None and (alone or pending_alone):
                Ft_y_total, Ft_ctf_total, stats = pending(Ft_y_total, Ft_ctf_total, stats)
                pending = None
            finish = _start_resident_local_chunk(
                chunk,
                tables=tables,
                posterior_bins=row_posterior_bin,
                experiment_dataset=data.experiment_dataset,
                bucket_io_kwargs=bucket_io_kwargs,
                fine_translation_prior_2d=fine_translation_prior_2d,
                half_weights=half_weights_windowed,
                full_to_compact=relion_score_full_to_compact,
                translation_angles=relion_score_translation_angles,
                n_score_pixels=int(n_windowed),
                mean=data.mean,
                volume_shape=volume_shape,
                disc_type=kernel.disc_type,
                projection_kwargs=projection_kwargs,
                projection_block_rows=projection_block_rows,
                projection_padding_factor=kernel.projection_padding_factor,
                relion_projector_half=relion_projector_half,
                relion_projector_r_max=kernel.relion_projector_r_max,
                precision_policy=precision_policy,
                n_fine_trans=n_fine_trans,
                n_recon_windowed=n_recon_windowed,
                n_rect=n_rect,
                mstep_block_rows=mstep_block_rows,
                adaptive_fraction=float(support.adaptive_fraction),
                keep_all_weights=not bool(support.reconstruct_significant_only),
                windowed_prepare=windowed_prepare,
                window_indices=window_indices,
                recon_window_indices=recon_window_indices,
                window_union=window_union,
                relion_x_half_recon_indices=relion_x_half_recon_indices,
                exact_positions_device=exact_positions_device,
                rect_indices_device=rect_indices_device,
                image_shape=image_shape,
                current_size=current_size,
                wsum_current_size=wsum_current_size,
                mstep_current_size=volume_current_size,
                mstep_max_r=mstep_adjoint_max_r(
                    volume_current_size, kernel.reconstruction_image_radius, kernel.reconstruction_padding_factor,
                    anisotropic_magnification=anisotropic_magnification,
                ),
                recon_volume_shape=recon_volume_shape,
                max_adjoint_block_bytes=max_adjoint_block_bytes,
                noise_variance_for_noise=noise_variance_for_noise_device,
                shell_indices_noise=shell_indices_noise_device,
                group_ids_np=group_ids_np,
                optics_groups_np=optics_groups_np,
                scale_corrections_np=scale_corrections_np,
                translation_prior_centers_np=translation_prior_centers_np,
                fine_translations=fine_translations,
                voxel_size=data.experiment_dataset.voxel_size,
                accumulate_noise=kernel.accumulate_noise,
                source_faithful_spectrum_norm=resolved_spectrum_norm,
                stats_config=stats_config,
                image_tables=image_tables,
                cuda_backproject=em_cuda_kernels,
                operand_route=operand_route,
                firstiter_cc=bool(kernel.firstiter_cc),
                relion_projector_capacity_texture=capacity_texture,
                class_projectors=class_projectors,
                class_scale_masks_rect=class_scale_masks_rect,
                lone_block_rows=lone_block_rows,
            )
            if pending is not None:
                Ft_y_total, Ft_ctf_total, stats = pending(Ft_y_total, Ft_ctf_total, stats)
            pending, pending_alone = finish, alone
            if not pipelined:
                Ft_y_total, Ft_ctf_total, stats = pending(Ft_y_total, Ft_ctf_total, stats)
                pending = None
        if pending is not None:
            Ft_y_total, Ft_ctf_total, stats = pending(Ft_y_total, Ft_ctf_total, stats)
        loop_s = time.time() - loop_t0
        if n_lone_chunks:
            logger.info(
                "Resident local pass-2 ran %d lone overflow image(s) in row blocks of %s rows",
                n_lone_chunks,
                "/".join(str(v) for v in sorted(lone_block_sizes, reverse=True)),
            )
    finally:
        for texture in class_textures:
            if texture is not None:
                texture.close()

    # ---- finalize ----------------------------------------------------------
    if tables.n_classes > 1:
        return _finalize_class_local_pass(
            Ft_y_total,
            Ft_ctf_total,
            stats,
            stats_config=stats_config,
            tables=tables,
            n_images=n_images,
            n_fine_trans=n_fine_trans,
            recon_volume_shape=recon_volume_shape,
            symmetry_label=symmetry_label,
            n_chunks=len(chunks),
            loop_s=loop_s,
            overall_t0=overall_t0,
        )
    # RELION symmetriseReconstructions (ml_optimiser.cpp:5541-5575): x=0
    # Hermitian enforcement, then applyPointGroupSymmetry on BPref.
    Ft_y_total, Ft_ctf_total = finalize_half_volume_bpref(
        Ft_y_total,
        Ft_ctf_total,
        recon_volume_shape,
        logger=logger,
        label="Resident local pass-2",
        symmetry_label=symmetry_label,
        relion_x_half=True,
    )
    Ft_y_total, Ft_ctf_total = relion_x_half_accumulators_to_public_layout(
        Ft_y_total,
        Ft_ctf_total,
        recon_volume_shape,
    )

    finalized = finalize_statistics(stats, config=stats_config, n_images=n_images)
    # ``best_fine_rotation_indices`` carries the winner's position in the
    # layout's flat row order, because that is what the chunk wrote into
    # ``best_fine_rot``. Every pose field is a lookup at that row.
    best_row = np.asarray(finalized.best_fine_rotation_indices, dtype=np.int64)
    best_translation = np.asarray(finalized.best_translation_indices, dtype=np.int64)
    if np.any(best_row < 0):
        raise RuntimeError("Resident local pass 2: an image has no winning candidate row")
    # RELION's hard assignment counts in the layout's own fine rotation ids,
    # as the retired exact local engine encoded it; the resident
    # statistics stage's image-local encoding is not that convention.
    hard_assignments = (
        tables.row_rotation_id[best_row].astype(np.int64) * np.int64(n_fine_trans)
        + best_translation
    )

    relion_stats = make_relion_stats(
        log_evidence_per_image=finalized.log_evidence_per_image,
        best_log_score_per_image=finalized.best_log_score_per_image,
        max_posterior_per_image=finalized.max_posterior_per_image,
        rotation_posterior_sums=_expand_posterior_bins(
            finalized.rotation_posterior_sums, posterior_bin_ids, tables.n_posterior_bins
        ),
    )
    noise_stats = make_noise_stats(
        wsum_sigma2_noise=finalized.wsum_sigma2_noise,
        wsum_img_power=finalized.wsum_img_power,
        wsum_sigma2_offset=finalized.wsum_sigma2_offset,
        sumw=finalized.sumw,
        wsum_norm_correction=finalized.wsum_norm_correction,
        wsum_scale_correction_xa=finalized.wsum_scale_correction_xa,
        wsum_scale_correction_aa=finalized.wsum_scale_correction_aa,
    )

    best_pose_rotations = best_pose_translations = best_pose_rotation_ids = None
    if support.return_best_pose_details:
        best_pose_rotations = np.asarray(tables.rotations)[best_row]
        best_pose_translations = np.asarray(tables.translation_grid)[best_translation]
        best_pose_rotation_ids = tables.row_rotation_id[best_row].astype(np.int64)
    best_pose_eulers_deg = (
        None if tables.source_eulers is None else tables.source_eulers[best_row]
    )

    profile = None
    if support.return_profile:
        profile = {
            "resident_local_chunks": np.int32(len(chunks)),
            "resident_local_rows": np.int64(tables.n_rows),
            "resident_local_loop_time_s": np.float64(loop_s),
            "resident_local_total_time_s": np.float64(time.time() - overall_t0),
            "resident_local_row_projection_bytes": np.int64(per_row_bytes),
            "resident_local_mstep_block_rows": np.int32(mstep_block_rows),
            "resident_local_projection_block_rows": np.int32(projection_block_rows),
        }

    logger.info(
        "Resident local pass-2 done: %d images, %d chunks, %.2fs chunk loop, %.2fs total",
        n_images,
        len(chunks),
        loop_s,
        time.time() - overall_t0,
    )
    return LocalEMResult(
        Ft_y=Ft_y_total,
        Ft_ctf=Ft_ctf_total,
        hard_assignments=hard_assignments,
        stats=relion_stats,
        best_pose_rotations=best_pose_rotations,
        best_pose_translations=best_pose_translations,
        best_pose_rotation_ids=best_pose_rotation_ids,
        noise_stats=noise_stats,
        profile=profile,
        best_pose_eulers_deg=best_pose_eulers_deg,
    )


def _finalize_class_local_pass(
    Ft_y_total,
    Ft_ctf_total,
    stats,
    *,
    stats_config,
    tables,
    n_images,
    n_fine_trans,
    recon_volume_shape,
    symmetry_label,
    n_chunks,
    loop_s,
    overall_t0,
):
    """A Class3D local pass's per-class BPrefs, statistics and winners, as ``ResidentKClassPass2Output``.

    The joint statistics (normalizer, Pmax, noise and scale sums) are over classes and poses; each
    class's evidence, winner and pruned M-step mass come from its sub-segments, and its BPref is
    symmetrised on its own, as the global K-class pass does (``compute_k_class_pass2_stats_resident``).
    Winners are rows of the layout's flat order, decoded through the tables as for K=1.
    """

    Ft_y_public, Ft_ctf_public = [], []
    for k in range(tables.n_classes):
        y, ctf = finalize_half_volume_bpref(
            Ft_y_total[k],
            Ft_ctf_total[k],
            recon_volume_shape,
            logger=logger,
            label=f"Resident local pass-2 class {k + 1}",
            symmetry_label=symmetry_label,
            relion_x_half=True,
        )
        y, ctf = relion_x_half_accumulators_to_public_layout(y, ctf, recon_volume_shape)
        Ft_y_public.append(y)
        Ft_ctf_public.append(ctf)

    finalized = finalize_statistics(stats, config=stats_config, n_images=n_images)
    if np.any(np.asarray(finalized.best_fine_rotation_indices) < 0):
        raise RuntimeError("Resident local pass 2: an image has no winning candidate row")
    per_class = finalized.classes
    class_row = np.asarray(per_class.best_fine_rotation_indices, dtype=np.int64)
    class_translation = np.asarray(per_class.best_translation_indices, dtype=np.int64)
    has_pose = class_row >= 0
    safe_row = np.where(has_pose, class_row, 0)
    safe_translation = np.where(has_pose, class_translation, 0)
    rotation_ids = tables.row_rotation_id[safe_row].astype(np.int64)
    rotation_posterior_sums = np.asarray(finalized.rotation_posterior_sums, dtype=np.float64)
    noise_stats = make_noise_stats(
        wsum_sigma2_noise=finalized.wsum_sigma2_noise,
        wsum_img_power=finalized.wsum_img_power,
        wsum_sigma2_offset=finalized.wsum_sigma2_offset,
        sumw=finalized.sumw,
        wsum_norm_correction=finalized.wsum_norm_correction,
        wsum_scale_correction_xa=finalized.wsum_scale_correction_xa,
        wsum_scale_correction_aa=finalized.wsum_scale_correction_aa,
    )
    rotations = np.asarray(tables.rotations)
    translations = np.asarray(tables.translation_grid)
    logger.info(
        "Resident local pass-2 done: %d images, %d classes, %d chunks, %.2fs chunk loop, %.2fs total",
        n_images,
        tables.n_classes,
        n_chunks,
        loop_s,
        time.time() - overall_t0,
    )
    return rp.ResidentKClassPass2Output(
        Ft_y=tuple(Ft_y_public),
        Ft_ctf=tuple(Ft_ctf_public),
        class_log_evidence_per_image=per_class.log_evidence,
        class_best_log_score_per_image=per_class.best_log_score,
        per_class_hard_assignments=np.where(
            has_pose, rotation_ids * np.int64(n_fine_trans) + safe_translation, -1
        ).astype(np.int64),
        stats=make_relion_stats(
            log_evidence_per_image=finalized.log_evidence_per_image,
            best_log_score_per_image=finalized.best_log_score_per_image,
            max_posterior_per_image=finalized.max_posterior_per_image,
            rotation_posterior_sums=np.sum(rotation_posterior_sums, axis=0),
        ),
        class_rotation_posterior_sums=rotation_posterior_sums,
        class_reconstruction_posterior_sums=per_class.posterior_sums,
        noise_stats=noise_stats,
        per_class_best_pose_rotations=tuple(rotations[safe_row[k]] for k in range(tables.n_classes)),
        per_class_best_pose_translations=tuple(translations[safe_translation[k]] for k in range(tables.n_classes)),
        per_class_best_pose_rotation_ids=tuple(rotation_ids[k] for k in range(tables.n_classes)),
        per_class_best_pose_eulers_deg=(
            None
            if tables.source_eulers is None
            else tuple(np.asarray(tables.source_eulers)[safe_row[k]] for k in range(tables.n_classes))
        ),
        profile={"resident_local_chunks": np.int32(n_chunks), "resident_local_loop_time_s": np.float64(loop_s)},
    )


# Probe chunks carry no reconstruction operands: a row keeps its score-window
# projection and its per-cell score/posterior outputs, an image its score operands.
_PROBE_IMAGE_CAPACITY_LADDER = (32, 128, 512)
_PROBE_CELL_BYTES = 6 * 4  # scores, probs, weights, reconstruction probs, mask, scratch


def _prepare_chunk_score_operands(
    *,
    chunk,
    image_indices,
    experiment_dataset,
    bucket_io_kwargs,
    windowed_prepare,
    score_window_indices,
    fine_translation_prior_2d,
    score_real_dtype,
    n_fine_trans,
    image_shape,
    current_size,
    source_faithful_spectrum_norm,
    normalized_cc=False,
):
    """One probe chunk's score operands, from the same preparation as pass 2.

    :func:`~relax.sparse_pass2.resident_pass2._prepare_chunk_reconstruction_operands`
    minus every reconstruction, noise and Wavg tile: ``_prepare_bucket_io`` runs
    with ``score_only`` and the rows are permuted and padded exactly as there.
    ``normalized_cc`` (the ``--firstiter_cc`` iteration) adds that function's
    normalized-CC operands, the translated corrected score tile and half the
    image power; ``corr_img_score`` is then the CC pixel weight.
    """

    image_indices = np.asarray(image_indices)
    batch_images, padded_ctf_params, fetched_indices, padded_fetched_indices = rp.fetch_capacity_batch(
        experiment_dataset, image_indices, chunk.image_capacity
    )
    order = rp._reorder_permutation(fetched_indices, image_indices, chunk.image_capacity)
    prepared = _prepare_bucket_io(
        experiment_dataset,
        batch_images,
        padded_ctf_params,
        padded_fetched_indices,
        return_direct_scoring_io=True,
        **bucket_io_kwargs,
    )
    batch_norm = prepared[2]
    ctf2_over_nv_half = prepared[3]
    processed_score_half_for_noise = prepared[6]
    shifted_corrected_score_half = prepared[7]
    direct_score_input = prepared[8]
    if windowed_prepare:
        score_input = direct_score_input
        corr_img_score = ctf2_over_nv_half
    else:
        gather_score = jnp.asarray(score_window_indices, dtype=jnp.int32)
        score_input = direct_score_input[:, gather_score]
        corr_img_score = ctf2_over_nv_half[:, gather_score]
    highres_xi2_half, _ = _relion_powerclass_noise_terms(
        processed_score_half_for_noise,
        image_shape=image_shape,
        current_size=current_size,
        use_exact_relion_gaussian=True,
        accumulate_noise=False,
        source_faithful_spectrum_norm=source_faithful_spectrum_norm,
    )
    permutation = jnp.asarray(order, dtype=jnp.int32)
    valid_images = jnp.asarray(np.arange(chunk.image_capacity) < chunk.n_valid_images, dtype=bool)

    def take(values):
        return rp._zero_padded_images(values[permutation], valid_images)

    translation_prior = jnp.asarray(
        np.zeros((chunk.image_capacity, int(n_fine_trans)), dtype=np.float32)
        if fine_translation_prior_2d is None
        else rp._pad_batch_to_capacity(np.asarray(fine_translation_prior_2d)[image_indices], chunk.image_capacity),
        dtype=score_real_dtype,
    )
    operands = {
        "score_input": take(score_input),
        "corr_img_score": take(corr_img_score),
        "highres_xi2_half": take(highres_xi2_half),
        "translation_prior": rp._zero_padded_images(translation_prior, valid_images),
    }
    if normalized_cc:
        tile = shifted_corrected_score_half.reshape(chunk.image_capacity, int(n_fine_trans), -1)
        if not windowed_prepare:
            tile = tile[:, :, jnp.asarray(score_window_indices, dtype=jnp.int32)]
        operands["score_shifted_cc"] = take(tile)
        operands["cc_half_batch_norm"] = take(0.5 * jnp.reshape(batch_norm, (chunk.image_capacity,)).real)
    return operands


def _cc_block_rows(row_capacity: int) -> int:
    """Rows per normalized-CC scoring block: the largest power of two up to 128 dividing the capacity."""

    block = 128
    while int(row_capacity) % block:
        block //= 2
    return block


def _local_firstiter_cc_posterior(
    scores, row_image_local, row_is_valid, segment_offsets, n_valid_images, *, image_capacity: int, cuda_backproject
):
    """RELION's ``--firstiter_cc`` posterior of a local chunk's normalized-CC scores.

    RELION zeroes every weight but the best one (ml_optimiser.cpp:9266-9292), so
    the posterior is one-hot at each image's first maximum in segment order and
    Pmax is 1, as in the global pass (``resident_pass2._winner_take_all_posterior``).
    log-Z is the log-sum-exp of the CC scores, the evidence the global pass reports.
    Returns ``(log_z, best_log_score, best_cell_index, max_posterior, row_posterior,
    n_significant)``.
    """

    scores = jnp.asarray(scores, dtype=jnp.float32)
    log_z = cuda_backproject.sparse_pass2_segmented_log_z_f64(scores.reshape(-1), segment_offsets, n_valid_images)
    best_log_score, best_cell_index, row_posterior = rp._winner_take_all_cells(
        scores, row_image_local, row_is_valid, segment_offsets, image_capacity=image_capacity
    )
    has_winner = jnp.isfinite(best_log_score)
    return (
        jnp.asarray(log_z, dtype=jnp.float64),
        best_log_score,
        best_cell_index,
        has_winner.astype(jnp.float32),
        row_posterior,
        has_winner.astype(jnp.int32),
    )


def _cap_significant_samples(mask, weights, row_bounds, max_significants: int):
    """RELION's ``maximum_significants`` cap on one chunk's pass-1 support.

    ``mask`` and ``weights`` are ``[rows, T]`` for the chunk's valid rows, and
    image ``i`` owns rows ``row_bounds[i]:row_bounds[i + 1]``. Where an image
    keeps more than ``max_significants`` samples, RELION lowers its count to the
    cap and takes the weight at that position of the ascending sort as the
    significant weight (acc_ml_optimiser_impl.h:3256-3263); every sample at or
    above it stays significant. The weights are the posterior's, whose order is
    RELION's sort key.
    """

    cap = int(max_significants)
    if cap <= 0:
        return mask
    mask = np.array(mask, dtype=bool, copy=True)
    weights = np.asarray(weights)
    for i in range(len(row_bounds) - 1):
        rs, re = int(row_bounds[i]), int(row_bounds[i + 1])
        image_mask = mask[rs:re]
        if int(np.count_nonzero(image_mask)) <= cap:
            continue
        image_weights = weights[rs:re]
        kept = image_weights[image_mask]
        significant_weight = np.partition(kept, kept.size - cap)[kept.size - cap]
        mask[rs:re] = image_mask & (image_weights >= significant_weight)
    return mask


def _probe_chunk_bytes(*, row_capacity: int, image_capacity: int, n_score_pixels: int, n_fine_trans: int,
                       complex_bytes: int, projection_transient_bytes: int) -> int:
    """Device bytes one probe chunk holds at its peak, with the previous chunk's outputs.

    A row keeps its score-window projection and its ``T`` posterior cells
    (scores, probabilities, weights, reconstruction probabilities, mask and
    scratch); an image keeps its score input, CTF operand and power terms. One
    projector call is live beside the rows while they are projected. The loop
    reads a chunk's support back after enqueuing the next chunk, so the previous
    chunk's cells are live too.
    """

    t = int(n_fine_trans)
    rows = int(row_capacity) * (int(n_score_pixels) * int(complex_bytes) + t * _PROBE_CELL_BYTES)
    images = int(image_capacity) * (int(n_score_pixels) * (2 * int(complex_bytes) + 8) + t * 4)
    previous = int(row_capacity) * t * _PROBE_CELL_BYTES
    return rows + images + int(projection_transient_bytes) + previous


def _run_resident_parent_probe(
    *,
    tables,
    experiment_dataset,
    bucket_io_kwargs,
    fine_translation_prior_2d,
    half_weights,
    full_to_compact,
    translation_angles,
    n_score_pixels,
    mean,
    volume_shape,
    disc_type,
    projection_kwargs,
    projection_padding_factor,
    relion_projector_half,
    relion_projector_r_max,
    relion_projector_capacity_texture,
    precision_policy,
    n_fine_trans,
    adaptive_fraction,
    windowed_prepare,
    window_indices,
    image_shape,
    current_size,
    source_faithful_spectrum_norm,
    return_profile,
    overall_t0,
    max_significants=-1,
    firstiter_cc=False,
    window_union=None,
    class_projectors=None,
) -> LocalEMResult:
    """RELION's local pass 1 (the adaptive parent probe) on the resident stages.

    Scores every parent candidate with the pass-2 scorer, forms the segmented
    float32 posterior with RELION's adaptive-fraction significance and its
    ``maximum_significants`` cap, and returns, per image, the significant
    samples as ``posterior_id * n_trans + t`` in the layout's row order: the
    ``reconstruction_sample_indices_by_image`` contract from which the local
    search builds pass 2's support.
    There is no M-step, so no accumulator, statistic or reconstruction operand
    is formed. A chunk's valid rows are projected into the score window alone
    (``window_union``, through the pass's staged texture), and its support is
    read back after the next chunk is enqueued, so the device scores one chunk
    while the host reads the previous one. RELION GPU sorts and sums the pass-1
    weights in float as well (acc_ml_optimiser_impl.h findSignificantPoints);
    a cutoff tie may resolve differently from the exact engine's float64 sort.

    A Class3D layout (``class_projectors``) projects each class's rows with its
    own reference; the significance is joint over the image's classes and poses,
    and a sample's posterior id carries its class (``class * n_bins + bin``).

    ``firstiter_cc`` scores the normalized CC and keeps only each image's best
    sample (RELION's ``--firstiter_cc`` pass 1: its binarized weights leave one
    significant coarse sample, ml_optimiser.cpp:9266-9292).
    """

    from relax.cuda import kernels as em_cuda_kernels

    t = int(n_fine_trans)
    complex_bytes = np.dtype(precision_policy.score_complex_dtype).itemsize
    projector_output_size = projection_kwargs.get("projector_output_size")
    n_half = int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)
    projection_row_bytes = rp.projection_call_row_bytes(
        crop_pixels=(
            n_half
            if projector_output_size is None
            else int(projector_output_size) * (int(projector_output_size) // 2 + 1)
        ),
        output_pixels=n_half if window_union is None else int(window_union.indices.shape[0]),
        n_score_pixels=int(n_score_pixels),
        n_recon_pixels=0,
        complex_bytes=int(np.dtype(relion_projector_half.dtype).itemsize),
    )
    budget = rp.resident_chunk_budget_bytes()
    projection_block_rows = _projection_block_rows(projection_row_bytes, budget)
    row_ladder = tuple(int(v) for v in parse_env_capacity_ladder(_ROW_CAPACITY_LADDER_ENV, _DEFAULT_ROW_CAPACITY_LADDER))
    image_ladder = tuple(int(v) for v in parse_env_capacity_ladder(_IMAGE_CAPACITY_LADDER_ENV, _PROBE_IMAGE_CAPACITY_LADDER))

    def peak(rows, images):
        return _probe_chunk_bytes(
            row_capacity=max(max(rows), rp.overflow_row_capacity(rp.max_image_rows(tables.row_offsets), rows)),
            image_capacity=max(images),
            n_score_pixels=int(n_score_pixels),
            n_fine_trans=t,
            complex_bytes=complex_bytes,
            projection_transient_bytes=min(projection_block_rows, max(rows)) * projection_row_bytes,
        )

    if budget is not None:
        while len(row_ladder) > 1 and peak(row_ladder, image_ladder) > budget:
            row_ladder = row_ladder[:-1]
        while len(image_ladder) > 1 and peak(row_ladder, image_ladder) > budget:
            image_ladder = image_ladder[:-1]
        if peak(row_ladder, image_ladder) > budget:
            raise ResidentConfigurationUnsupported(
                "The device-resident local pass-1 parent probe does not implement this configuration: "
                f"its smallest chunk needs {peak(row_ladder, image_ladder) / 1024**3:.2f} "
                f"GiB against a {budget / 1024**3:.2f} GiB budget"
            )
    chunks = plan_local_capacity_chunks(tables, row_capacity_ladder=row_ladder, image_capacity_ladder=image_ladder)
    logger.info(
        "Resident local pass-1 probe plan: %d images, %d candidate rows, %d translations -> %d chunks "
        "(row capacities %s, image capacities %s, projection block rows %d, projection window %s px); "
        "chunk peak %.2f GiB of a %s budget",
        tables.n_images,
        tables.n_rows,
        t,
        len(chunks),
        ",".join(str(v) for v in row_ladder),
        ",".join(str(v) for v in image_ladder),
        projection_block_rows,
        "full" if window_union is None else int(window_union.indices.shape[0]),
        peak(row_ladder, image_ladder) / float(1024**3),
        rp.format_budget_gib(budget),
    )

    sample_ids_by_image: list[np.ndarray] = [np.zeros(0, dtype=np.int64)] * tables.n_images
    log_evidence = np.zeros(tables.n_images, dtype=np.float64)
    best_log_score = np.zeros(tables.n_images, dtype=np.float64)
    max_posterior = np.zeros(tables.n_images, dtype=np.float64)
    hard_assignments = np.full(tables.n_images, -1, dtype=np.int64)

    def start(chunk):
        """Enqueue one chunk up to its posterior; returns what :func:`finish` reads back."""

        image_indices = np.arange(chunk.image_start, chunk.image_stop, dtype=np.int64)
        host_chunk = materialize_local_chunk(tables, chunk)
        ops = _prepare_chunk_score_operands(
            chunk=chunk,
            image_indices=image_indices,
            experiment_dataset=experiment_dataset,
            bucket_io_kwargs=bucket_io_kwargs,
            windowed_prepare=windowed_prepare,
            score_window_indices=window_indices,
            fine_translation_prior_2d=fine_translation_prior_2d,
            score_real_dtype=precision_policy.score_real_dtype,
            n_fine_trans=t,
            image_shape=image_shape,
            current_size=current_size,
            source_faithful_spectrum_norm=source_faithful_spectrum_norm,
            normalized_cc=bool(firstiter_cc),
        )
        # Host rotations: the coarse kernel's wrapped-row check reads each
        # block's first rotation, which on a device array waits for the queue.
        projection_options = dict(
            score_indices=window_indices,
            recon_indices=None,
            max_projected_rotations=int(projection_block_rows),
            output_complex_dtype=precision_policy.score_complex_dtype,
            output_abs2_dtype=precision_policy.score_real_dtype,
            relion_projector_r_max=relion_projector_r_max,
            projection_padding_factor=projection_padding_factor,
            window_union=window_union,
            **projection_kwargs,
        )
        if class_projectors is None:
            score_proj, _, _, _ = project_resident_live_rows(
                mean,
                np.asarray(host_chunk["rotations"], dtype=np.dtype(precision_policy.score_real_dtype)),
                image_shape,
                volume_shape,
                disc_type,
                n_valid_rows=int(chunk.n_valid_rows),
                relion_projector_half=relion_projector_half,
                relion_projector_capacity_texture=relion_projector_capacity_texture,
                **projection_options,
            )
        else:
            (score_proj, _, _), _ = _project_class_rows(
                host_chunk,
                class_projectors,
                image_shape=image_shape,
                volume_shape=volume_shape,
                disc_type=disc_type,
                n_valid_rows=int(chunk.n_valid_rows),
                rotation_dtype=precision_policy.score_real_dtype,
                projection_options=projection_options,
                host_rotations=True,
            )
        n_valid_images_device = jnp.asarray(host_chunk["n_valid_images"], dtype=jnp.int32)
        chunk_image_ids = jnp.where(
            jnp.arange(chunk.image_capacity, dtype=jnp.int32) < n_valid_images_device,
            jnp.arange(chunk.image_capacity, dtype=jnp.int32),
            jnp.int32(-1),
        )
        segment_offsets_np = chunk_segment_offsets(tables, chunk, n_fine_trans=t)
        segment_offsets = jnp.asarray(segment_offsets_np, dtype=jnp.int32)
        if firstiter_cc:
            row_image_local = jnp.asarray(host_chunk["row_image_local"], dtype=jnp.int32)
            scored = score_resident_projected_chunk_normalized_cc(
                score_proj,
                row_image_local,
                None if host_chunk["row_mask_bits"] is None else jnp.asarray(host_chunk["row_mask_bits"], dtype=jnp.uint8),
                jnp.asarray(host_chunk["n_valid_rows"], dtype=jnp.int32),
                ops["score_shifted_cc"],
                ops["corr_img_score"],
                ops["cc_half_batch_norm"],
                half_weights=half_weights,
                full_to_compact=full_to_compact,
                row_capacity=int(chunk.row_capacity),
                n_fine_trans=t,
                block_rows=_cc_block_rows(int(chunk.row_capacity)),
            )
            del score_proj, ops
            log_z_out, best_log, best_cell, max_post, winner, n_significant = _local_firstiter_cc_posterior(
                scored.scores,
                row_image_local,
                jnp.arange(int(chunk.row_capacity), dtype=jnp.int32) < int(chunk.n_valid_rows),
                segment_offsets,
                n_valid_images_device,
                image_capacity=chunk.image_capacity,
                cuda_backproject=em_cuda_kernels,
            )
            weights = winner.reshape(-1)
            mask = weights > 0
        else:
            scored = score_resident_projected_chunk(
                score_proj,
                jnp.asarray(host_chunk["row_image_local"], dtype=jnp.int32),
                jnp.asarray(host_chunk["row_log_prior"], dtype=jnp.float32),
                None if host_chunk["row_mask_bits"] is None else jnp.asarray(host_chunk["row_mask_bits"], dtype=jnp.uint8),
                jnp.asarray(host_chunk["n_valid_rows"], dtype=jnp.int32),
                chunk_image_ids,
                ops["score_input"],
                ops["corr_img_score"],
                ops["highres_xi2_half"],
                ops["translation_prior"],
                half_weights=half_weights,
                translation_angles=translation_angles,
                full_to_compact=full_to_compact,
                logical_current_size=jnp.asarray(current_size, dtype=jnp.int32),
                row_capacity=int(chunk.row_capacity),
                image_capacity=chunk.image_capacity,
                n_fine_trans=t,
                n_score_pixels=int(n_score_pixels),
            )
            del score_proj, ops
            scores_flat = jnp.asarray(scored.scores, dtype=jnp.float32).reshape(-1)
            log_z = em_cuda_kernels.sparse_pass2_segmented_log_z_f64(
                scores_flat, segment_offsets, n_valid_images_device
            )
            (
                log_z_out,
                best_log,
                best_cell,
                max_post,
                _probs,
                weights,
                _reconstruction_probs,
                mask,
                n_significant,
                _sum_weight,
                _threshold,
            ) = em_cuda_kernels.sparse_pass2_segmented_posterior_f32(
                scores_flat,
                segment_offsets,
                n_valid_images_device,
                log_z,
                jnp.ones((chunk.image_capacity,), dtype=jnp.float32),
                adaptive_fraction=float(adaptive_fraction),
                keep_all=False,
                use_external_sum_weight=False,
            )
        device = (mask, best_cell, log_z_out, best_log, max_post, n_significant, weights, scored.min_diff2)
        return chunk, host_chunk, segment_offsets_np, device

    def finish(chunk, host_chunk, segment_offsets_np, device):
        """Read one chunk's support back and record it per image."""

        mask, best_cell, log_z_out, best_log, max_post, n_significant, weights, min_diff2 = device
        mask_np, best_cell_np, log_z_np, best_log_np, max_post_np, n_sig_np, min_diff2_np = jax.device_get(
            (mask, best_cell, log_z_out, best_log, max_post, n_significant, min_diff2)
        )
        mask_np = np.asarray(mask_np, dtype=bool).reshape(chunk.row_capacity, t)[:chunk.n_valid_rows]
        image_row_start = segment_offsets_np.astype(np.int64)[:chunk.n_valid_images] // t
        image_n_sig = np.asarray(n_sig_np)[: chunk.n_valid_images]
        if max_significants > 0 and int(np.max(image_n_sig, initial=0)) > max_significants:
            weights_np = np.asarray(jax.device_get(weights), dtype=np.float32).reshape(chunk.row_capacity, t)
            weights_np = weights_np[: chunk.n_valid_rows]
            row_bounds = np.append(image_row_start, chunk.n_valid_rows)
            mask_np = _cap_significant_samples(mask_np, weights_np, row_bounds, max_significants)
        rows, cols = np.nonzero(mask_np)
        posterior_ids = host_chunk["row_posterior_id"][:chunk.n_valid_rows].astype(np.int64)
        sample_ids = posterior_ids[rows] * np.int64(t) + cols.astype(np.int64)
        row_image = host_chunk["row_image_local"][:chunk.n_valid_rows][rows]
        # np.nonzero is row-major, and an image's rows are contiguous, so each
        # image's samples come out in the exact engine's (row, translation) order.
        bounds = np.searchsorted(row_image, np.arange(chunk.n_valid_images + 1))
        for local in range(chunk.n_valid_images):
            image = int(chunk.image_start) + local
            sample_ids_by_image[image] = sample_ids[bounds[local] : bounds[local + 1]]
            best_row = image_row_start[local] + int(best_cell_np[local]) // t
            hard_assignments[image] = (
                int(host_chunk["row_rotation_id"][best_row]) * t + int(best_cell_np[local]) % t
            )
        sl = slice(int(chunk.image_start), int(chunk.image_stop))
        # The scores are centred on RELION's common minimum; the evidences are reported absolute,
        # as pass 2 reports them (its log_score_offset is -min_diff2).
        min_diff2_np = np.asarray(min_diff2_np, dtype=np.float64)[:chunk.n_valid_images]
        log_evidence[sl] = np.asarray(log_z_np, dtype=np.float64)[:chunk.n_valid_images] - min_diff2_np
        best_log_score[sl] = np.asarray(best_log_np, dtype=np.float64)[:chunk.n_valid_images] - min_diff2_np
        max_posterior[sl] = np.asarray(max_post_np)[:chunk.n_valid_images]

    loop_t0 = time.time()
    pending = None
    for chunk in chunks:
        started = start(chunk)
        if pending is not None:
            finish(*pending)
        pending = started
    if pending is not None:
        finish(*pending)
    loop_s = time.time() - loop_t0

    if any(ids.size == 0 for ids in sample_ids_by_image):
        raise RuntimeError("Resident local pass-1 probe: an image kept no significant sample")
    stats = make_relion_stats(
        log_evidence_per_image=log_evidence,
        best_log_score_per_image=best_log_score,
        max_posterior_per_image=max_posterior,
        rotation_posterior_sums=np.zeros(0, dtype=np.float64),
        host_arrays=True,
    )
    profile = {
        "reconstruction_sample_indices_by_image": tuple(sample_ids_by_image),
        "resident_probe_chunks": np.int32(len(chunks)),
        "resident_probe_loop_time_s": np.float64(loop_s),
        "resident_probe_total_time_s": np.float64(time.time() - overall_t0),
    }
    logger.info(
        "Resident local pass-1 probe done: %d images, %d chunks, %d significant samples, "
        "%.2fs chunk loop, %.2fs total",
        tables.n_images,
        len(chunks),
        int(sum(ids.size for ids in sample_ids_by_image)),
        loop_s,
        time.time() - overall_t0,
    )
    return LocalEMResult(
        Ft_y=None,
        Ft_ctf=None,
        hard_assignments=hard_assignments,
        stats=stats,
        profile=profile,
    )


def _unshifted_chunk_operands(experiment_dataset, image_indices, **kwargs):
    """:func:`resident_pass2.unshifted_chunk_operands`, or ``None`` where they are refused."""

    try:
        return rp.unshifted_chunk_operands(experiment_dataset, image_indices, **kwargs)
    except ResidentOperandsUnsupported as reason:
        logger.info(
            "Resident local pass-2 keeps the pre-shifted translation tiles: %s", reason
        )
        return None


class _LiveRowsFirst(NamedTuple):
    """A chunk's M-step rows, live rows first (see :func:`_live_rows_first`)."""

    row_ids: jax.Array  # int32 [C_R]: the chunk row at each M-step position
    row_image_local: jax.Array  # int32 [C_R]
    kernel_row_image_ids: jax.Array  # int32 [C_R]
    row_posterior: jax.Array  # float32 [C_R, T]


@jax.jit
def _live_rows_first(row_posterior, row_is_valid, row_image_local, kernel_row_image_ids):
    """Order a chunk's rows live first (stable) and count the live ones.

    A row is live when it is valid and has a positive reconstruction weight.
    The M-step sums are taken in a different row grouping than chunk order;
    each row's contribution is unchanged.
    """

    live = row_is_valid & jnp.any(row_posterior > 0, axis=1)
    order = jnp.argsort(jnp.where(live, 0, 1).astype(jnp.int32), stable=True).astype(jnp.int32)
    rows = _LiveRowsFirst(
        row_ids=order,
        row_image_local=row_image_local[order],
        kernel_row_image_ids=kernel_row_image_ids[order],
        row_posterior=row_posterior[order],
    )
    return rows, jnp.sum(live, dtype=jnp.int32)


def _open_capacity_texture(
    relion_projector_half,
    *,
    relion_projector_r_max,
    projection_padding_factor,
    projection_kwargs,
):
    """One staged projector texture for a half's chunks, or None where it does not apply.

    Every chunk projects the same slab, and the per-call projector stages its
    texture and synchronizes the stream each time
    (:func:`~relax.cuda.kernels.project_relion_half_capacity`). Where that
    half-storage kernel is the one the projection takes
    (:func:`~relax.helpers.projection.relion_capacity_texture_serves`), the
    texture is staged once here instead; the caller closes it after the loop.
    """

    from relax.helpers.projection import relion_capacity_texture_serves

    if relion_projector_half is None or jax.default_backend() != "gpu":
        return None
    if not relion_capacity_texture_serves(
        relion_projector_half,
        r_max=int(relion_projector_r_max),
        padding_factor=int(projection_padding_factor),
        projector_output_size=int(projection_kwargs["projector_output_size"]),
        relion_texture_interp=projection_kwargs.get("relion_texture_interp"),
    ):
        return None
    from relax.cuda.kernels import RelionCapacityHalfTextureF32

    return RelionCapacityHalfTextureF32(
        relion_projector_half,
        int(relion_projector_r_max),
        padding_factor=int(projection_padding_factor),
    )


# The K=1 rotation posterior's used-bin count changes with every local pass, and the
# statistics programs are shaped by it: padded to a multiple of this, a refinement's
# passes share a few shapes instead of compiling the image-term program per half.
_POSTERIOR_BIN_QUANTUM = 4096


def _posterior_bin_capacity(n_used_bins: int, *, n_classes: int) -> int:
    """The accumulator length of ``n_used_bins`` used posterior bins (K>1 keeps the whole histogram)."""

    if n_classes != 1:
        return int(n_used_bins)
    return max(1, -(-int(n_used_bins) // _POSTERIOR_BIN_QUANTUM)) * _POSTERIOR_BIN_QUANTUM


def _expand_posterior_bins(sums, posterior_bin_ids, n_posterior_bins: int) -> np.ndarray:
    """The rotation posterior over the layout's whole histogram, from its used bins (host float64).

    ``sums`` may be padded past the used bins (:func:`_posterior_bin_capacity`); the padding holds no mass.
    """

    used = np.asarray(posterior_bin_ids, dtype=np.int64)
    dense = np.zeros(int(n_posterior_bins), dtype=np.float64)
    dense[used] = np.asarray(jax.device_get(sums), dtype=np.float64)[: used.size]
    return dense


class _ClassRows(NamedTuple):
    """The two fields of a chunk's rows :func:`resident_pass2._class_sub_segment_posterior` reads."""

    classes: object  # resident_pass2._ChunkClassLayout
    n_valid_images: object  # int32 device scalar


def _sum_class_mstep_terms(
    class_terms, class_scale_masks_rect, *, class_layout, class_posterior, row_start, row_capacity, n_fine_trans
):
    """A Class3D chunk's M-step terms summed over classes, and the class fields of its statistics.

    ``class_terms`` are each class's ``(wavg triplet, noise shells, A2, XA)`` from its own M-step. The
    noise residual and the norm terms add over classes; each class's Wavg XA/AA enter the scale sums
    only under its own ``data_vs_prior_class > 3`` mask (acc_ml_optimiser_impl.h:4893-4912), so they are
    folded here and cleared from the triplet, as ``resident_pass2._fold_class_scale_sums`` does for
    the global pass. A class's winner is a row of the layout's flat order, ``row * T + t``.
    """

    zero = jnp.float32(0.0)
    triplet = noise_shells = a2 = xa = scale_xa = scale_aa = None
    for k, (triplet_k, noise_k, a2_k, xa_k) in enumerate(class_terms):
        mask = jnp.asarray(class_scale_masks_rect[k], dtype=bool).reshape(1, -1)
        xa_scale = jnp.sum(jnp.where(mask, triplet_k[:, :, 0], zero).astype(jnp.float64), axis=1)
        aa_scale = jnp.sum(jnp.where(mask, triplet_k[:, :, 1], zero).astype(jnp.float64), axis=1)
        diff2 = triplet_k.at[:, :, :2].set(zero)
        if triplet is None:
            triplet, noise_shells, a2, xa, scale_xa, scale_aa = diff2, noise_k, a2_k, xa_k, xa_scale, aa_scale
        else:
            triplet, noise_shells = triplet + diff2, noise_shells + noise_k
            a2, xa = a2 + a2_k, xa + xa_k
            scale_xa, scale_aa = scale_xa + xa_scale, scale_aa + aa_scale
    t = jnp.int64(int(n_fine_trans))
    class_best_row = jnp.clip(
        class_layout.segment_row_start + class_posterior.best_cell_index // t, 0, jnp.int64(max(row_capacity - 1, 0))
    )
    class_fields = dict(
        row_class=class_layout.row_class,
        per_class_log_z=class_posterior.log_z,
        per_class_best_log_score=class_posterior.best_log_score,
        per_class_best_cell=jnp.where(
            jnp.isfinite(class_posterior.best_log_score),
            (jnp.int64(int(row_start)) + class_best_row) * t + class_posterior.best_cell_index % t,
            jnp.int64(-1),
        ),
        scale_xa_per_image=scale_xa,
        scale_aa_per_image=scale_aa,
    )
    return triplet, noise_shells, a2, xa, class_fields


def _project_class_rows(
    host_chunk,
    class_projectors,
    *,
    image_shape,
    volume_shape,
    disc_type,
    n_valid_rows: int,
    rotation_dtype,
    projection_options: dict,
    host_rotations: bool = False,
):
    """Project a Class3D chunk's rows, each class's with its own reference, into the chunk's row order.

    Rows stay image-major then class-major, as the joint posterior needs, so a class's rows are not
    contiguous: each class's rotations are gathered to the front of a capacity-sized array and its
    projections written back at their rows (``place_rows`` of :func:`project_resident_live_rows`).
    ``host_rotations`` hands the projector host rotations, as the pass-1 probe does.
    Returns ``((score_proj, recon_proj, recon_abs2), n_projected_rows)``.
    """

    row_capacity = int(np.shape(host_chunk["rotations"])[0])
    row_class = np.asarray(host_chunk["row_class"])[: int(n_valid_rows)]
    outputs = None
    n_projected = 0
    for k, projector in enumerate(class_projectors):
        rows = np.flatnonzero(row_class == k).astype(np.int32)
        if rows.size == 0:
            continue
        rotations = np.broadcast_to(np.eye(3, dtype=np.dtype(rotation_dtype)), (row_capacity, 3, 3)).copy()
        rotations[: rows.size] = np.asarray(host_chunk["rotations"])[rows]
        place = np.arange(row_capacity, 2 * row_capacity, dtype=np.int32)
        place[: rows.size] = rows
        score_proj, recon_proj, recon_abs2, n_class_projected = project_resident_live_rows(
            projector.mean,
            rotations if host_rotations else jnp.asarray(rotations, dtype=rotation_dtype),
            image_shape,
            volume_shape,
            disc_type,
            n_valid_rows=int(rows.size),
            relion_projector_half=projector.slab,
            relion_projector_capacity_texture=projector.texture,
            place_rows=place,
            outputs=outputs,
            **projection_options,
        )
        outputs = (score_proj, recon_proj, recon_abs2)
        n_projected += int(n_class_projected)
    if outputs is None:
        raise RuntimeError("a Class3D local chunk has no valid row")
    return outputs, n_projected


def _start_resident_local_chunk(
    chunk,
    *,
    tables,
    posterior_bins,
    experiment_dataset,
    bucket_io_kwargs,
    fine_translation_prior_2d,
    half_weights,
    full_to_compact,
    translation_angles,
    n_score_pixels,
    mean,
    volume_shape,
    disc_type,
    projection_kwargs,
    projection_block_rows,
    projection_padding_factor,
    relion_projector_half,
    relion_projector_r_max,
    precision_policy,
    n_fine_trans,
    n_recon_windowed,
    n_rect,
    mstep_block_rows,
    adaptive_fraction,
    keep_all_weights,
    windowed_prepare,
    window_indices,
    recon_window_indices,
    window_union,
    relion_x_half_recon_indices,
    exact_positions_device,
    rect_indices_device,
    image_shape,
    current_size,
    mstep_current_size,
    recon_volume_shape,
    wsum_current_size=None,
    mstep_max_r=None,
    max_adjoint_block_bytes,
    noise_variance_for_noise,
    shell_indices_noise,
    group_ids_np,
    scale_corrections_np,
    translation_prior_centers_np,
    fine_translations,
    voxel_size,
    accumulate_noise,
    source_faithful_spectrum_norm,
    optics_groups_np=None,
    stats_config,
    image_tables,
    cuda_backproject,
    operand_route,
    firstiter_cc=False,
    relion_projector_capacity_texture=None,
    class_projectors=None,
    class_scale_masks_rect=None,
    lone_block_rows=None,
):
    """Enqueue one local capacity chunk's front stages; return its ``finish``.

    The front is the chunk's operands, projections, scores, segmented
    posterior and live-row order, none of which reads the half's running
    accumulators. ``finish(Ft_y_total, Ft_ctf_total, stats)`` reads the live row
    count back, runs the M-step and the image statistics and returns the
    updated accumulators, so the caller can enqueue the next chunk's front
    before it finishes this one.

    ``operand_route`` is the half's mutable choice of reconstruction operands:
    ``{"unshifted": True}`` prepares the chunk's unshifted per-image operands
    and lets the M-step translate them inside T15's translate-and-sum kernel,
    as the global pass does (T16); ``False`` keeps the pre-shifted
    ``[images, translations, pixels]`` tiles and the XLA reduction. A
    configuration the unshifted preparation refuses switches the half to the
    tiles at its first chunk.

    ``firstiter_cc`` scores the chunk's rows with RELION's normalized CC and puts each image's
    whole posterior on its best cell (:func:`_local_firstiter_cc_posterior`).

    The only host work inside is the chunk's operand upload, the T7 offsets
    readback the segmented posterior performs internally, the optional
    significance-count pull and ``finish``'s live row count; no per-chunk
    result is otherwise brought back.

    ``lone_block_rows`` is set for one image's overflow chunk: its rows are projected and scored that many at a
    time, keeping only their ``[C_R, T]`` scores (:func:`~relax.sparse_pass2.resident_scoring.score_resident_chunk_in_row_blocks`,
    shared with the global pass), the posterior is formed over all of them at once, and each M-step block projects
    its own rows. Each row's projection and score are those of the one-call chunk; only the grouping of the
    projector calls changes. In a Class3D pass each block's rows project with their own class's reference (the
    lone image's rows are class-major, so a block spans few classes) and each class's M-step blocks with its own.
    The ``--firstiter_cc`` iteration scores the blocks with the normalized-CC block core
    (:func:`~relax.sparse_pass2.resident_scoring.score_resident_chunk_normalized_cc_in_row_blocks`).
    """

    if lone_block_rows is not None:
        if int(lone_block_rows) % int(mstep_block_rows) or int(chunk.row_capacity) % int(lone_block_rows):
            raise ValueError(
                f"lone row blocks of {lone_block_rows} must divide the chunk's {chunk.row_capacity} rows and be a "
                f"multiple of the M-step's {mstep_block_rows}-row blocks"
            )
    image_indices = np.arange(chunk.image_start, chunk.image_stop, dtype=np.int64)
    # The operands' powerClass terms sum above the weighted sums' size (--strict_highres_exp).
    operand_current_size = current_size if wsum_current_size is None else wsum_current_size

    profile = parse_env_flag(_CHUNK_PROFILE_ENV, default=False)
    marks: dict[str, float] = {}

    def mark(name, *values):
        if not profile:
            return
        if values:
            jax.block_until_ready(values)
        marks[name] = time.time()

    mark("t0")
    host_chunk = materialize_local_chunk(tables, chunk)
    row_image_local = jnp.asarray(host_chunk["row_image_local"], dtype=jnp.int32)
    row_log_prior = jnp.asarray(host_chunk["row_log_prior"], dtype=jnp.float32)
    row_mask_bits = (
        None
        if host_chunk["row_mask_bits"] is None
        else jnp.asarray(host_chunk["row_mask_bits"], dtype=jnp.uint8)
    )
    image_ids = jnp.asarray(host_chunk["image_ids"], dtype=jnp.int32)
    n_valid_rows_device = jnp.asarray(host_chunk["n_valid_rows"], dtype=jnp.int32)
    n_valid_images_device = jnp.asarray(host_chunk["n_valid_images"], dtype=jnp.int32)
    row_is_valid = jnp.arange(chunk.row_capacity, dtype=jnp.int32) < n_valid_rows_device
    kernel_row_image_ids = jnp.where(row_is_valid, row_image_local, jnp.int32(-1))

    # --- stage 0: this chunk's operands, both families, at image capacity ---
    # Upstream merged the per-half score preparation into this call, so
    # ``_prepare_bucket_io`` runs once per image instead of twice and every
    # per-chunk program is keyed on the image-capacity class rather than on
    # the chunk's occupancy.
    recon = None
    if operand_route["unshifted"]:
        recon = _unshifted_chunk_operands(
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
            accumulate_noise=accumulate_noise,
            source_faithful_spectrum_norm=bool(source_faithful_spectrum_norm),
            fine_translation_prior_2d=fine_translation_prior_2d,
            scale_corrections_np=scale_corrections_np,
            group_ids_np=group_ids_np,
            optics_groups_np=optics_groups_np,
            precision_policy=precision_policy,
        )
        if recon is None:
            operand_route["unshifted"] = False
    if recon is None:
        recon = rp._prepare_chunk_reconstruction_operands(
            chunk=chunk,
            image_indices=image_indices,
            experiment_dataset=experiment_dataset,
            bucket_io_kwargs=bucket_io_kwargs,
            windowed_prepare=windowed_prepare,
            recon_window_indices=recon_window_indices,
            n_fine_trans=int(n_fine_trans),
            n_recon_windowed=int(n_recon_windowed),
            image_shape=image_shape,
            current_size=operand_current_size,
            use_exact_relion_gaussian=True,
            accumulate_noise=accumulate_noise,
            source_faithful_spectrum_norm=bool(source_faithful_spectrum_norm),
            score_window_indices=window_indices,
            fine_translation_prior_2d=fine_translation_prior_2d,
            score_real_dtype=precision_policy.score_real_dtype,
            relion_score_translation_angles=translation_angles,
            rect_indices_device=rect_indices_device,
            exact_positions_device=exact_positions_device,
            scale_corrections_np=scale_corrections_np,
            group_ids_np=group_ids_np,
            optics_groups_np=optics_groups_np,
            noise_shell_indices_half=image_tables.shell_indices_half,
            n_noise_shells=int(stats_config.n_shells),
            normalized_cc=bool(firstiter_cc),
        )
    recon_operand = recon["recon_image"] if recon.get("recon_image") is not None else recon["shifted_recon"]

    mark("operands", recon_operand, recon["score_input"])

    # --- stages 1-2: project this chunk's own rows -------------------------
    # Only the valid rows: the padding past them (about 70% of a chunk at the
    # 10097 local iterations) is never scored or reconstructed.
    projection_options = dict(
        score_indices=window_indices,
        recon_indices=recon_window_indices,
        max_projected_rotations=int(projection_block_rows),
        output_complex_dtype=precision_policy.score_complex_dtype,
        output_abs2_dtype=precision_policy.score_real_dtype,
        relion_projector_r_max=relion_projector_r_max,
        projection_padding_factor=projection_padding_factor,
        window_union=window_union,
        **projection_kwargs,
    )
    chunk_rotations = jnp.asarray(host_chunk["rotations"], dtype=precision_policy.score_real_dtype)

    def project_rows(rotations, n_valid_rows, projector=None):
        """Rows projected with the pass's reference, or with one Class3D class's ``projector``."""

        if projector is not None:
            return project_resident_live_rows(
                projector.mean,
                rotations,
                image_shape,
                volume_shape,
                disc_type,
                n_valid_rows=n_valid_rows,
                relion_projector_half=projector.slab,
                relion_projector_capacity_texture=projector.texture,
                **projection_options,
            )
        return project_resident_live_rows(
            mean,
            rotations,
            image_shape,
            volume_shape,
            disc_type,
            n_valid_rows=n_valid_rows,
            relion_projector_half=relion_projector_half,
            relion_projector_capacity_texture=relion_projector_capacity_texture,
            **projection_options,
        )

    def project_score_block(start, stop):
        """Score-window projections of chunk rows ``start:stop``, each row with its own class's reference."""

        n_valid = max(0, min(int(chunk.n_valid_rows), stop) - start)
        if class_projectors is None:
            return project_rows(chunk_rotations[start:stop], n_valid)[0]
        if n_valid == 0:
            return jnp.zeros((stop - start, int(n_score_pixels)), dtype=precision_policy.score_complex_dtype)
        (score_block, _, _), _ = _project_class_rows(
            {"rotations": np.asarray(host_chunk["rotations"])[start:stop],
             "row_class": np.asarray(host_chunk["row_class"])[start:stop]},
            class_projectors,
            image_shape=image_shape,
            volume_shape=volume_shape,
            disc_type=disc_type,
            n_valid_rows=n_valid,
            rotation_dtype=precision_policy.score_real_dtype,
            projection_options=projection_options,
        )
        return score_block

    if lone_block_rows is not None:
        score_proj = recon_proj = recon_abs2 = None
        n_projected_rows = int(chunk.n_valid_rows)
    elif class_projectors is None:
        score_proj, recon_proj, recon_abs2, n_projected_rows = project_rows(chunk_rotations, chunk.n_valid_rows)
    else:
        (score_proj, recon_proj, recon_abs2), n_projected_rows = _project_class_rows(
            host_chunk,
            class_projectors,
            image_shape=image_shape,
            volume_shape=volume_shape,
            disc_type=disc_type,
            n_valid_rows=chunk.n_valid_rows,
            rotation_dtype=precision_policy.score_real_dtype,
            projection_options=projection_options,
        )

    mark("project", score_proj, recon_proj, recon_abs2)

    segment_offsets_np = chunk_segment_offsets(tables, chunk, n_fine_trans=n_fine_trans)
    segment_offsets = jnp.asarray(segment_offsets_np, dtype=jnp.int32)
    if firstiter_cc:
        # --- stages 3-4, --firstiter_cc: normalized CC, winner takes all ---
        if lone_block_rows is not None:

            def cc_block_reference(start):
                return project_score_block(start, min(start + int(lone_block_rows), int(chunk.row_capacity)))

            scored = score_resident_chunk_normalized_cc_in_row_blocks(
                cc_block_reference,
                row_image_local,
                row_mask_bits,
                int(chunk.n_valid_rows),
                recon["score_shifted_cc"],
                recon["corr_img_score"],
                recon["cc_half_batch_norm"],
                block_rows=int(lone_block_rows),
                half_weights=half_weights,
                full_to_compact=full_to_compact,
                row_capacity=chunk.row_capacity,
                n_fine_trans=int(n_fine_trans),
            )
        else:
            scored = score_resident_projected_chunk_normalized_cc(
                score_proj,
                row_image_local,
                row_mask_bits,
                n_valid_rows_device,
                recon["score_shifted_cc"],
                recon["corr_img_score"],
                recon["cc_half_batch_norm"],
                half_weights=half_weights,
                full_to_compact=full_to_compact,
                row_capacity=chunk.row_capacity,
                n_fine_trans=int(n_fine_trans),
                block_rows=_cc_block_rows(chunk.row_capacity),
            )
        del score_proj
        scores_flat = jnp.asarray(scored.scores, dtype=jnp.float32).reshape(-1)
        mark("score", scores_flat)
        (
            log_z_out,
            best_log_score,
            best_cell_index,
            max_posterior,
            row_posterior,
            n_significant,
        ) = _local_firstiter_cc_posterior(
            scored.scores,
            row_image_local,
            row_is_valid,
            segment_offsets,
            n_valid_images_device,
            image_capacity=chunk.image_capacity,
            cuda_backproject=cuda_backproject,
        )
    else:
        # --- stage 3: score ----------------------------------------------------
        # Chunk-local image slots address the chunk's own operands.
        chunk_image_ids = jnp.where(
            jnp.arange(chunk.image_capacity, dtype=jnp.int32) < n_valid_images_device,
            jnp.arange(chunk.image_capacity, dtype=jnp.int32),
            jnp.int32(-1),
        )
        if lone_block_rows is not None:

            def block_reference(start):
                # The projection window union covers both windows, so the block projects both; the planned
                # largest-class chunk counts both.
                return project_score_block(start, min(start + int(lone_block_rows), int(chunk.row_capacity)))

            scored = score_resident_chunk_in_row_blocks(
                block_reference,
                row_image_local,
                row_log_prior,
                expand_local_chunk_mask_jnp(row_mask_bits, n_trans=int(n_fine_trans)),
                int(chunk.n_valid_rows),
                chunk_image_ids,
                recon["score_input"],
                recon["corr_img_score"],
                recon["highres_xi2_half"],
                recon["translation_prior"],
                block_rows=int(lone_block_rows),
                half_weights=half_weights,
                translation_angles=translation_angles,
                full_to_compact=full_to_compact,
                logical_current_size=jnp.asarray(current_size, dtype=jnp.int32),
                row_capacity=chunk.row_capacity,
                image_capacity=chunk.image_capacity,
                n_fine_trans=int(n_fine_trans),
            )
        else:
            scored = score_resident_projected_chunk(
                score_proj,
                row_image_local,
                row_log_prior,
                row_mask_bits,
                n_valid_rows_device,
                chunk_image_ids,
                recon["score_input"],
                recon["corr_img_score"],
                recon["highres_xi2_half"],
                recon["translation_prior"],
                half_weights=half_weights,
                translation_angles=translation_angles,
                full_to_compact=full_to_compact,
                logical_current_size=jnp.asarray(current_size, dtype=jnp.int32),
                row_capacity=chunk.row_capacity,
                image_capacity=chunk.image_capacity,
                n_fine_trans=int(n_fine_trans),
                n_score_pixels=int(n_score_pixels),
            )
        del score_proj
        scores_flat = jnp.asarray(scored.scores, dtype=jnp.float32).reshape(-1)
        mark("score", scores_flat)

        # --- stage 4: segmented RELION float32 fine posterior -------------------
        log_z = cuda_backproject.sparse_pass2_segmented_log_z_f64(
            scores_flat, segment_offsets, n_valid_images_device
        )
        posterior = cuda_backproject.sparse_pass2_segmented_posterior_f32(
            scores_flat,
            segment_offsets,
            n_valid_images_device,
            log_z,
            jnp.ones((chunk.image_capacity,), dtype=jnp.float32),
            adaptive_fraction=float(adaptive_fraction),
            keep_all=bool(keep_all_weights),
            use_external_sum_weight=False,
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
            n_significant,
            _sum_weight,
            _threshold,
        ) = posterior
        row_posterior = jnp.asarray(reconstruction_probs, dtype=jnp.float32).reshape(
            chunk.row_capacity, int(n_fine_trans)
        )
    mark("posterior", row_posterior, log_z_out, best_cell_index)

    mstep_rotations = jnp.asarray(
        host_chunk["mstep_rotations"], dtype=precision_policy.score_real_dtype
    )

    # Only rows the pruned posterior keeps enter the M-step, live rows first in
    # chunk order, as in the global pass (_make_mstep_block_inputs): a row with
    # no positive cell adds exact zeros to every M-step accumulator, and the
    # block walk stops after the live rows. At the 10097 full-box final pass
    # about a third of the scored rows carry no weight.
    class_layout = class_posterior = None
    if class_projectors is None:
        mstep_rows, n_live_rows = _live_rows_first(
            row_posterior, row_is_valid, row_image_local, kernel_row_image_ids
        )
    else:
        # Class3D: the image's joint posterior above, each (image, class) sub-segment's
        # evidence and winner here, and each class's live rows for its own BPref.
        n_classes = len(class_projectors)
        class_layout = rp._chunk_class_layout(
            host_chunk, chunk, n_classes=n_classes, n_fine_trans=int(n_fine_trans), place=rp._PLACE_ON_DEVICE
        )
        class_posterior = rp._class_sub_segment_posterior(
            scores_flat.reshape(chunk.row_capacity, int(n_fine_trans)),
            _ClassRows(classes=class_layout, n_valid_images=n_valid_images_device),
            row_is_valid,
            n_segments=chunk.image_capacity * n_classes,
            n_classes=n_classes,
            cuda_backproject=cuda_backproject,
        )
        # A class's M-step reads only its own rows' weights: the block walk's last block runs past the
        # live rows, so the other classes' rows there must carry zero posterior, not theirs. Each class's
        # order is formed when its M-step runs, so one class's copy of the posterior lives at a time.
        def class_mstep_rows(k):
            in_class = class_layout.row_class == k
            return _live_rows_first(
                jnp.where(in_class[:, None], row_posterior, jnp.float32(0.0)),
                row_is_valid & in_class,
                row_image_local,
                kernel_row_image_ids,
            )

        mstep_rows = n_live_rows = None

    def finish(Ft_y_total, Ft_ctf_total, stats):
        """The chunk's M-step and statistics, added into the running accumulators."""

        n_live_rows_host = int(n_live_rows) if class_projectors is None else 0  # Class3D: summed over classes below

        # P3-G (default on): the three chunk-wide arrays go to the M-step entry
        # point whole and the block program gathers its rows inside the jit.
        # RELAX_LOCAL_SEARCH_RESIDENT_BLOCK_ROW_PROGRAM=0 has the Python callback
        # gather them per block, three eager dispatches each time; on the full
        # EMPIAR-10097 run that path took 2964 s against 2868 s (job 14550046).
        block_row_program = parse_env_flag(_BLOCK_ROW_PROGRAM_ENV, default=True)

        def run_mstep(rows, n_live, Ft_y, Ft_ctf, projector=None):
            if lone_block_rows is not None:
                # Each M-step block projects its own (live-first) rows: the lone image's projections never all
                # live at once. A Class3D class's blocks hold its own rows (the others carry zero posterior), so
                # they project with that class's reference.
                chunk_projections = None

                def block_projections(start, stop):
                    block_rows = rows.row_ids[start:stop]
                    _, recon_block, abs2_block, _ = project_rows(
                        chunk_rotations[block_rows], int(stop - start), projector
                    )
                    return recon_block, abs2_block, mstep_rotations[block_rows]

            elif block_row_program:
                block_projections = None
                chunk_projections = (recon_proj, recon_abs2, mstep_rotations)
            else:
                chunk_projections = None

                def block_projections(start, stop):
                    block_rows = rows.row_ids[start:stop]
                    return recon_proj[block_rows], recon_abs2[block_rows], mstep_rotations[block_rows]

            return rp.run_resident_mstep_blocks(
                block_projections,
                chunk_projections=chunk_projections,
                row_capacity=chunk.row_capacity,
                n_valid_rows=n_live,
                mstep_block_rows=int(mstep_block_rows),
                image_capacity=chunk.image_capacity,
                row_image_local=rows.row_image_local,
                kernel_row_image_ids=rows.kernel_row_image_ids,
                row_posterior=rows.row_posterior,
                row_ids=rows.row_ids,
                recon=recon,
                recon_pixel_indices=recon_window_indices,
                translation_angles=translation_angles,
                n_rect=int(n_rect),
                n_shells=int(stats_config.n_shells),
                n_recon_windowed=int(n_recon_windowed),
                noise_variance_for_noise=noise_variance_for_noise,
                shell_indices_noise=shell_indices_noise,
                exact_positions_device=exact_positions_device,
                Ft_y_total=Ft_y,
                Ft_ctf_total=Ft_ctf,
                image_shape=image_shape,
                recon_volume_shape=recon_volume_shape,
                mstep_current_size=mstep_current_size,
                mstep_max_r=mstep_max_r,
                relion_x_half_recon_indices=relion_x_half_recon_indices,
                max_adjoint_block_bytes=max_adjoint_block_bytes,
                cuda_backproject=cuda_backproject,
                n_optics_groups=int(stats_config.n_optics_groups),
            )

        class_fields = {}
        if class_projectors is None:
            (
                Ft_y_total,
                Ft_ctf_total,
                wavg_triplet_pixels,
                block_noise_shells,
                a2_per_image,
                xa_per_image,
            ) = run_mstep(mstep_rows, n_live_rows_host, Ft_y_total, Ft_ctf_total)
        else:
            Ft_y_total, Ft_ctf_total = list(Ft_y_total), list(Ft_ctf_total)
            class_terms = []
            for k, projector in enumerate(class_projectors):
                rows_k, n_live_device = class_mstep_rows(k)
                n_live_k = int(n_live_device)
                n_live_rows_host += n_live_k
                Ft_y_total[k], Ft_ctf_total[k], *terms = run_mstep(
                    rows_k, n_live_k, Ft_y_total[k], Ft_ctf_total[k], projector
                )
                del rows_k
                class_terms.append(terms)
            wavg_triplet_pixels, block_noise_shells, a2_per_image, xa_per_image, class_fields = _sum_class_mstep_terms(
                class_terms,
                class_scale_masks_rect,
                class_layout=class_layout,
                class_posterior=class_posterior,
                row_start=int(chunk.row_start),
                row_capacity=chunk.row_capacity,
                n_fine_trans=int(n_fine_trans),
            )

        mark("mstep", Ft_y_total, Ft_ctf_total, wavg_triplet_pixels, block_noise_shells)

        # --- stage 7: image-level statistics ------------------------------------
        translation_sqdist_ang = image_tables.translation_sqdist_ang
        if translation_prior_centers_np is not None:
            from relax.helpers.translation_prior import (
                translation_prior_centers_for_images,
                translation_sqdist_angstrom,
            )

            # Build the centres at capacity on the host so the squared-distance
            # program is keyed on the capacity class, not the occupancy; padded
            # rows multiply a zero posterior, so their value is never observable.
            padded_image_indices = rp._pad_batch_to_capacity(
                np.asarray(image_indices).reshape(-1, 1), chunk.image_capacity
            ).reshape(-1)
            centers = translation_prior_centers_for_images(
                translation_prior_centers_np,
                padded_image_indices,
                batch_size=chunk.image_capacity,
            )
            translation_sqdist_ang = rp._zero_padded_images(
                jnp.asarray(translation_sqdist_angstrom(fine_translations, centers, voxel_size)),
                jnp.asarray(np.arange(chunk.image_capacity) < chunk.n_valid_images, dtype=bool),
            )
        chunk_tables = image_tables._replace(translation_sqdist_ang=translation_sqdist_ang)

        image_row_start_np = segment_offsets_np.astype(np.int64)[:chunk.image_capacity] // int(n_fine_trans)
        image_row_count_np = (
            segment_offsets_np.astype(np.int64)[1:] - segment_offsets_np.astype(np.int64)[:-1]
        ) // int(n_fine_trans)
        image_row_start = jnp.asarray(image_row_start_np, dtype=jnp.int64)
        image_row_count = jnp.asarray(image_row_count_np, dtype=jnp.int64)

        best_row_local = jnp.asarray(best_cell_index, dtype=jnp.int64) // jnp.int64(n_fine_trans)
        slot_is_valid = jnp.arange(chunk.image_capacity, dtype=jnp.int32) < n_valid_images_device
        invalid_best = slot_is_valid & ((best_row_local < 0) | (best_row_local >= image_row_count))
        best_chunk_row = jnp.clip(
            image_row_start + best_row_local, 0, jnp.int64(max(chunk.row_capacity - 1, 0))
        ).astype(jnp.int32)
        # The winner's global row in the layout's flat order; the pose decode reads
        # rotations, M-step rotations, fine ids and source Eulers at that row.
        best_global_row = jnp.int64(int(chunk.row_start)) + best_chunk_row.astype(jnp.int64)

        chunk_posterior_bin = np.full(chunk.row_capacity, int(stats_config.n_coarse_rot), dtype=np.int32)
        chunk_posterior_bin[:chunk.n_valid_rows] = posterior_bins[int(chunk.row_start) : int(chunk.row_stop)]
        chunk_posterior_bin = jnp.asarray(chunk_posterior_bin)
        chunk_operands = rp._ChunkImageOperands(
            row_posterior=row_posterior,
            row_image_local=row_image_local,
            row_coarse_rot=jnp.where(
                row_is_valid, chunk_posterior_bin, jnp.int32(int(stats_config.n_coarse_rot))
            ),
            image_ids=image_ids,
            group_ids=recon["group_ids"],
            image_power_shells=recon["image_power_shells"],
            relion_norm_high_shell=recon["relion_norm_high_shell"],
            wavg_triplet_pixels=wavg_triplet_pixels,
            block_noise_shells=block_noise_shells,
            a2_per_image=a2_per_image,
            xa_per_image=xa_per_image,
            class_log_z=jnp.asarray(log_z_out, dtype=jnp.float64),
            min_diff2=scored.min_diff2,
            best_log_score=best_log_score,
            max_posterior=max_posterior,
            best_cell_index=jnp.asarray(best_cell_index, dtype=jnp.int64),
            best_fine_rot=best_global_row,
            optics_groups=recon.get("optics_groups"),
            **class_fields,
        )
        stats = rp._accumulate_chunk_image_terms(
            stats, chunk_operands, chunk_tables, config=stats_config
        )
        stats = stats._replace(
            invalid_best_rows=stats.invalid_best_rows + jnp.sum(invalid_best.astype(jnp.int64))
        )
        mark("stats", stats)
        if profile:
            order = ("t0", "operands", "project", "score", "posterior", "mstep", "stats")
            spans = {
                name: marks[name] - marks[prev]
                for prev, name in zip(order, order[1:])
                if name in marks and prev in marks
            }
            n_blocks = len(range(0, min(chunk.row_capacity, max(n_live_rows_host, 1)), int(mstep_block_rows))) or 1
            logger.info(
                "Resident local chunk profile: images=%d/%d rows=%d/%d live_rows=%d row_pad=%.1f%% "
                "blocks=%d proj_rows=%d recon_tile=%s wavg_tile=%s | %s | chunk=%.3fs",
                chunk.n_valid_images, chunk.image_capacity, chunk.n_valid_rows, chunk.row_capacity, n_live_rows_host,
                100.0 * (chunk.row_capacity - chunk.n_valid_rows) / max(chunk.row_capacity, 1),
                n_blocks, n_projected_rows,
                f"{recon_operand.dtype}{tuple(recon_operand.shape)}",
                f"{recon['raw_translated_wavg_rectangle'].dtype}"
                f"{tuple(recon['raw_translated_wavg_rectangle'].shape)}",
                " ".join(f"{k}={v:.3f}s" for k, v in spans.items()),
                marks["stats"] - marks["t0"],
            )
        return Ft_y_total, Ft_ctf_total, stats

    return finish
