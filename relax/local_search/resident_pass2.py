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
site in ``relax/refinement/local_half.py``). Any other configuration raises
:class:`NotImplementedError` naming the missing piece: there is no other local
fine-pass engine to fall back to.

The pass-1 parent probe
-----------------------
``score_only`` with ``return_reconstruction_sample_indices`` runs RELION's local
pass 1 on the same stages (:func:`~relax.local_search.resident_parent_probe.run_resident_parent_probe`): score-only
operands, rows projected into the score window alone, the segmented float32
posterior with the adaptive fraction and RELION's ``maximum_significants`` cap
(``resident_parent_probe._cap_significant_samples``), and per-image significant samples in the
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
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from recovar.reconstruction import noise as noise_utils

from relax.fine_pass import (
    projection_blocks,
    resident_chunk_memory,
    resident_chunk_records,
    resident_image_terms,
    resident_pass_tables,
    resident_stable_windows,
    resident_statistics,
)
from relax.fine_pass import resident_pass2 as rp
from relax.fine_pass.bucket_io import (
    relion_cuda_score_translation_angles_if_available,
)
from relax.fine_pass.policy import (
    ResidentConfigurationUnsupported,
)
from relax.fine_pass.projection_blocks import (
    projection_kwargs_for_relion_score_window,
)
from relax.fine_pass.resident_operands import (
    ResidentOperandsUnsupported,
    require_unshifted_operand_support,
)
from relax.fine_pass.resident_scoring import (
    resident_row_projection_bytes,
)
from relax.fine_pass.window import (
    pass2_half_weights,
    pass2_window_setup,
    sparse_pass2_window_setup,
)
from relax.fourier.preprocessing import half_translation_phase_table
from relax.local_search import (
    records,
    resident_local_admission,
    resident_local_chunk,
    resident_parent_probe,
)
from relax.local_search.resident_layout import (
    DEFAULT_IMAGE_CAPACITY_LADDER,
    DEFAULT_ROW_CAPACITY_LADDER,
    plan_local_capacity_chunks,
    tables_from_local_layout,
)
from relax.reconstruction.half_volume_mstep import (
    allocate_row_partial,
    half_volume_accumulator_shape,
    relion_x_half_mstep_accumulator_dtypes,
)
from relax.relion.optics_aberrations import dataset_magnification_is_anisotropic, dataset_needs_exact_ctf
from relax.relion.projector_setup import (
    cast_relion_projector_for_execution,
    prepare_local_class_projector_slabs,
    prepare_local_projector_slab,
)
from relax.relion.scale_groups import prepare_scale_correction_groups
from relax.runtime.memory_budget import (
    max_adjoint_block_bytes_for_pass,
    max_translation_tile_bytes_for_pass,
    projection_cache_max_bytes_for_pass,
)
from relax.types import LocalEMResult, make_relion_stats

logger = logging.getLogger(__name__)


__all__ = [
    "compute_local_search_resident",
]


class _ClassProjector(NamedTuple):
    """One class's reference for a Class3D local pass: the volume, its slab (or its shape) and texture."""

    mean: object
    slab: object
    texture: object


def _local_chunk_loop_pipelined(*, chunk_profile: bool) -> bool:
    """Whether the chunk loop enqueues chunk k+1 before chunk k's M-step (off while profiling: ``chunk_profile``,
    ``EngineTuning.local_chunk_profile``).

    The chunk plan counts the second chunk's arrays exactly when it does
    (:func:`resident_chunk_memory.resident_chunk_bytes`).
    """

    return not chunk_profile


def _unshifted_operands_route(bucket_io_kwargs, *, window_indices, recon_window_indices, requested: bool) -> bool:
    """Whether this pass's chunks take the unshifted per-image operands (T16) or the translated tiles
    (``requested``: ``EngineTuning.resident_operands``)."""

    if not requested:
        return False
    try:
        require_unshifted_operand_support(
            bucket_io_kwargs, window_indices=window_indices, recon_window_indices=recon_window_indices
        )
    except ResidentOperandsUnsupported as reason:
        logger.info("Resident local pass-2 keeps the pre-shifted translation tiles: %s", reason)
        return False
    return True


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
    data: records.LocalSearchData,
    local_layout,
    kernel: records.LocalSearchKernelPolicy,
    support: records.LocalSearchSupportPolicy,
    *,
    translation_prior_centers,
    symmetry_label: str,
) -> LocalEMResult:
    """Run one K=1 local-search fine pass 2 on the device-resident stages.

    ``data`` holds the half's images, reference and per-image corrections, ``local_layout`` the
    per-image candidate rows (:mod:`relax.local_search.layout`), ``kernel`` the window, projector and
    optics choices and ``support`` the posterior support and what is returned
    (:mod:`relax.local_search.records`). ``translation_prior_centers`` is each image's
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
    as the global ``--firstiter_cc`` pass does (resident_chunk_posterior ``_resident_chunk_posterior_firstiter_cc``).
    In a Class3D the best hidden variable is over the classes of the layout too: RELION scores every
    class in its CC iteration when the references are given, and class 1 alone only when it generates
    seeds from one reference (``do_generate_seeds``, ml_optimiser.cpp:4392-4401), a start whose layout
    the caller has restricted to that class (``restrict_local_layout_classes``).
    """

    from recovar import cuda_backproject

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
        ) = pass2_window_setup(
            image_shape,
            current_size=current_size,
            reconstruction_current_size=kernel.reconstruction_current_size,
            half_spectrum_scoring=kernel.half_spectrum_scoring,
            square_window=kernel.square_window,
            relion_firstiter_score_mode=score_mode,
            use_exact_relion_gaussian=not kernel.firstiter_cc,
            use_float64_scoring=kernel.use_float64_scoring,
            device_memory_override=support.engine_tuning.device_memory_bytes,
            # RELION's window at every size, including the box (a shape class reaches its
            # box before the reference does): the resident driver never scores a full half.
            window_at_box=True,
            reference_sphere_clip=kernel.reconstruction_image_radius is not None or anisotropic_magnification,
        )
    except NotImplementedError as exc:
        # Reported before any device work, like resident_local_admission.require_resident_local_configuration.
        raise ResidentConfigurationUnsupported(str(exc)) from exc

    # The retired exact local engine used this flag as passed rather than resolving it
    # against the environment, so do the same: it selects RELION's powerClass
    # shell spectrum for the image-power statistics and, with it, the
    # deterministic float64 norm reduction. It does not switch on RELION's
    # exact BPref operands here, because the exact local engine never enables
    # those on its production path.
    resolved_spectrum_norm = bool(kernel.source_faithful_spectrum_norm)
    scale_groups_available = data.group_ids is not None
    relion_wavg_atomic_scale_aa = resident_statistics.relion_wavg_atomic_triplet_runs(
        accumulate_noise=kernel.accumulate_noise, scale_groups_available=scale_groups_available
    )
    resident_local_admission.require_resident_local_configuration(
        requested_diagnostics=support.engine_tuning.resident_diagnostics,
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
    )
    resident_local_admission.require_local_support(
        jax.default_backend() == "gpu" and cuda_backproject.custom_cuda_requested(),
        "every resident stage is a CUDA FFI target",
    )
    # Stable windows (relax#95), as the global resident pass runs them and with its exclusions (the
    # --firstiter_cc iteration, a size at the box, a weighted-sum size above the score's): the pass runs
    # in its physical window class, so the logical sizes of a class share the chunk programs.
    stable_window_plan, stable_window_spec, program_current_size = resident_stable_windows.pass_stable_windows(
        image_shape,
        current_size=current_size,
        wsum_current_size=wsum_current_size,
        mstep_current_size=mstep_current_size,
        n_half=n_half,
        square_window=kernel.square_window,
        window_spec_kwargs=window_spec_kwargs,
        firstiter_cc=kernel.firstiter_cc,
        requested=support.engine_tuning.stable_windows,
        quantum=support.engine_tuning.stable_window_quantum,
        log_label="Resident local pass-2",
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
    window_setup = sparse_pass2_window_setup(
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
        half_spectrum_scoring=kernel.half_spectrum_scoring,
        relion_firstiter_score_mode=score_mode,
        use_float64_scoring=kernel.use_float64_scoring,
        nyquist_column_counting=kernel.nyquist_column_counting,
    )
    del half_weights
    half_weights_windowed, relion_score_full_to_compact = resident_stable_windows.score_weights_and_lookup(
        half_weights_windowed,
        image_shape=image_shape,
        current_size=current_size,
        window_indices_np=window_indices_np,
        plan=stable_window_plan,
    )
    # The exact local engine casts sigma2 to the score real dtype before it
    # divides by it (local_big_jit.py, the non-BPref-operand branch). Passing a
    # float64 sigma2 through instead makes every downstream operand double:
    # ctf^2/sigma2 becomes float64, the translated reconstruction tile becomes
    # complex128, and prepare_bucket_io then picks relion_translate_score_f64
    # where the exact engine picks the f32 kernel. That is a different
    # computation and twice the memory on the largest per-chunk array.
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(
        data.noise_variance, image_shape
    ).squeeze().astype(precision_policy.score_real_dtype)
    n_optics_groups, optics_groups_np = resident_pass_tables.optics_group_rows(
        noise_variance_half, data.optics_group_ids, n_images=n_images
    )
    relion_score_translation_angles = relion_cuda_score_translation_angles_if_available(
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
    # Class3D masks each class's scale sums by its own curve; the per-image tables take class 1's.
    class_masks_np = resident_pass_tables.scale_pixel_masks_rect(
        data.scale_correction_data_vs_prior,
        n_classes=tables.n_classes,
        shell_indices_noise=shell_indices_noise,
        n_shells=n_shells,
        relion_wavg_rectangle=relion_wavg_rectangle,
    )
    scale_pixel_mask_rect_np = class_masks_np[0]
    class_scale_masks_rect = None if tables.n_classes == 1 else jnp.asarray(class_masks_np)
    group_ids_np, n_scale_groups = prepare_scale_correction_groups(
        data.group_ids, data.scale_correction_group_count, n_images=n_images
    )

    # ---- accumulator layout (identical to the exact engine's x-half BPref) -
    volume_current_size, recon_volume_shape, mstep_max_r = resident_pass_tables.reconstruction_volume(
        volume_shape,
        mstep_current_size=mstep_current_size,
        reconstruction_volume_current_size=kernel.reconstruction_volume_current_size,
        reconstruction_padding_factor=kernel.reconstruction_padding_factor,
        reconstruction_image_radius=kernel.reconstruction_image_radius,
        anisotropic_magnification=anisotropic_magnification,
    )
    # The BPref cube the chunks accumulate into: RELION's, or the plan's physical one, cropped at finalize.
    program_volume_current_size, program_recon_volume_shape, program_mstep_max_r = (
        resident_stable_windows.physical_reconstruction_volume(
            stable_window_plan,
            volume_shape=volume_shape,
            volume_current_size=volume_current_size,
            reconstruction_volume_current_size=kernel.reconstruction_volume_current_size,
            recon_volume_shape=recon_volume_shape,
            mstep_max_r=mstep_max_r,
            reconstruction_image_radius=kernel.reconstruction_image_radius,
            reconstruction_padding_factor=kernel.reconstruction_padding_factor,
            anisotropic_magnification=anisotropic_magnification,
            quantum=support.engine_tuning.stable_window_quantum,
        )
    )
    recon_accum_shape = half_volume_accumulator_shape(program_recon_volume_shape)
    recon_volume_size = int(np.prod(recon_accum_shape))
    recon_y_accum_dtype, recon_ctf_accum_dtype = relion_x_half_mstep_accumulator_dtypes(
        data.experiment_dataset.dtype,
        use_relion_x_half_mstep=True,
        mstep_double=support.scoring_overrides.x_half_mstep_double,
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
    projection_kwargs = projection_kwargs_for_relion_score_window(
        window_spec.projection_kwargs(return_abs2=False),
        use_relion_projector=True,
        current_size=current_size,
    )
    # The exact local engine passes all three of these to the same projector;
    # ``relion_texture_interp=None`` means "resolve as strict parity does":
    # the run's ScoringOverrides.projector_texture_interp (RELION's CUDA texture interpolator by default).
    projection_kwargs["relion_texture_interp"] = kernel.projection_relion_texture_interp
    projection_kwargs["texture_interp_default"] = support.scoring_overrides.projector_texture_interp
    projection_kwargs["force_jax"] = False
    projection_kwargs["mask_current_image_disk"] = False
    projection_kwargs["relion_kernel"] = kernel.projection_relion_kernel
    # A stable-window pass projects at its physical class (pass_capacity_projection): center-padded slabs,
    # the static radius sentinel 0 and RELION's logical model and image radii at runtime, through the
    # pass's staged texture, which holds the padded slab at the logical model radius.
    projection_halves, projection_r_max, capacity_projection_kwargs, capacity_logical_output_size = (
        resident_stable_windows.pass_capacity_projection(
            stable_window_plan,
            use_relion_projector=True,
            class_projector_halves=[relion_projector_half] if class_slabs is None else class_slabs,
            relion_projector_r_max=kernel.relion_projector_r_max,
            projection_padding_factor=kernel.projection_padding_factor,
            projection_kwargs=projection_kwargs,
        )
    )
    texture_logical_r_max = None
    if capacity_logical_output_size is not None:
        projection_kwargs["projector_output_size"] = int(stable_window_plan.physical_current_size)
        projection_kwargs.update(capacity_projection_kwargs)
        relion_projector_half = projection_halves[0]
        if class_slabs is not None:
            class_slabs = projection_halves
        texture_logical_r_max = int(kernel.relion_projector_r_max)
    # The slabs live on in relion_projector_half and class_slabs alone, which the pass releases once their
    # textures are staged; a list kept here held the device slab for the whole pass (relax#108: the 2.71 GiB
    # box-448 slab beside the final pass's BPref at a 16 GB device).
    del projection_halves
    # The pixels the projector computes for a chunk's rows: the two windows,
    # not the full half spectrum (10202 box 800: ~155k of 320,800 px per row).
    window_union = (
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

    # One RELION projector texture for the whole pass, staged before the
    # capacity plan so the plan's device reading sees it. Projected per call,
    # the texture path stages the slab's two float32 CUDA arrays outside the
    # XLA pool on every chunk (7.9 GB at EMPIAR-10202 current size 626, which
    # failed with CUDA out of memory once the pool had grown, bigbox 14480549,
    # 14507538) and synchronizes the stream. RELION builds its projector
    # texture once per iteration.
    capacity_texture = projection_blocks.open_pass_capacity_texture(
        relion_projector_half,
        relion_projector_r_max=kernel.relion_projector_r_max,
        projection_padding_factor=kernel.projection_padding_factor,
        projection_kwargs=projection_kwargs,
        class_index=0,
        logical_r_max=texture_logical_r_max,
    )
    class_textures = [capacity_texture]
    try:
        if class_slabs is not None:
            class_textures += [
                projection_blocks.open_pass_capacity_texture(
                    slab,
                    relion_projector_r_max=kernel.relion_projector_r_max,
                    projection_padding_factor=kernel.projection_padding_factor,
                    projection_kwargs=projection_kwargs,
                    class_index=k,
                    logical_r_max=texture_logical_r_max,
                )
                for k, slab in enumerate(class_slabs[1:], start=1)
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
            finite_check_mode=support.engine_tuning.finite_check,
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
            # significance only, no M-step and no accumulators. See resident_parent_probe.run_resident_parent_probe.
            return resident_parent_probe.run_resident_parent_probe(
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
                relion_projector_r_max=projection_r_max,
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
                engine_tuning=support.engine_tuning,
                window_union=(
                    resident_stable_windows.pass_window_union(
                        window_indices,
                        image_shape=image_shape,
                        projector_output_size=int(projection_kwargs["projector_output_size"]),
                        capacity_logical_output_size=capacity_logical_output_size,
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
        # Every local row is backprojected on its own: the rows go through one partial pair (relax#65).
        row_partial = allocate_row_partial(
            recon_volume_size,
            recon_y_accum_dtype,
            recon_ctf_accum_dtype,
            stock_relion_direct_accumulation=support.scoring_overrides.stock_relion_direct_accumulation,
        )

        # The operand family decides what a chunk holds, so it is chosen before the
        # plan; the per-chunk fallback in resident_local_chunk.start_resident_local_chunk stays as a guard.
        # The --firstiter_cc iteration scores the translated corrected score tile, which only
        # the pre-shifted family prepares (resident_pass2 refuses unshifted operands for it too).
        operand_route = {
            "unshifted": (not kernel.firstiter_cc) and _unshifted_operands_route(
                bucket_io_kwargs, window_indices=window_indices, recon_window_indices=recon_window_indices,
                requested=support.engine_tuning.resident_operands,
            )
        }

        # ---- capacity plan ----------------------------------------------------
        row_ladder = _cap_row_capacity_ladder(
            support.engine_tuning.local_row_capacities or DEFAULT_ROW_CAPACITY_LADDER,
            n_score_pixels=n_windowed,
            n_recon_pixels=n_recon_windowed,
            max_bytes=projection_cache_max_bytes_for_pass(
                device_memory_bytes, override=support.engine_tuning.projection_max_bytes,
            ),
        )
        # The accumulators already exist, so the budget reading counts them; the
        # plan records them so that its pass_bytes is the pass's total need.
        accumulator_bytes = tables.n_classes * resident_chunk_memory.resident_accumulator_bytes(
            recon_volume_size, recon_y_accum_dtype, recon_ctf_accum_dtype
        )
        chunk_budget_bytes = resident_chunk_memory.resident_chunk_budget_bytes()
        # One projector call holds the texture crop and the gathered rows
        # together (resident_chunk_memory.projection_call_row_bytes); its block of rows is sized to
        # the per-call cap and its bytes enter the chunk plan.
        projector_output_size = projection_kwargs.get("projector_output_size")
        projector_slab_bytes = int(np.dtype(relion_projector_half.dtype).itemsize)
        projection_row_bytes = resident_chunk_memory.projection_call_row_bytes(
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
        projection_block_rows = resident_local_chunk.projection_block_rows(
            projection_row_bytes, chunk_budget_bytes, support.engine_tuning.local_projection_call_max_bytes
        )
        tile_pixels = resident_chunk_memory.chunk_translated_tile_pixels(
            unshifted_operands=operand_route["unshifted"],
            n_score_pixels=n_windowed,
            n_recon_pixels=n_recon_windowed,
            n_rect_pixels=n_rect,
            n_exact_rect_pixels=int(relion_wavg_rectangle.exact_positions.size),
            normalized_cc=bool(kernel.firstiter_cc),
            masked_scoring=True,
        )
        image_ladder = support.engine_tuning.local_image_capacities or DEFAULT_IMAGE_CAPACITY_LADDER
        if not (operand_route["unshifted"] and chunk_budget_bytes is not None):
            # The fixed translation-tile budget bounds the pre-shifted route's
            # three recon tiles per image. An unshifted chunk holds only the
            # Wavg rectangle and its exact positions, which the joint plan below
            # counts against the measured budget; capped at three recon tiles,
            # the 10097 full-box final pass ran 32-image chunks, 2032 per half.
            image_ladder = resident_chunk_memory.resident_image_capacity_start(
                image_ladder,
                n_fine_trans=n_fine_trans,
                n_recon_pixels=n_recon_windowed,
                max_tile_bytes=max_translation_tile_bytes_for_pass(
                    device_memory_bytes, override=support.engine_tuning.translation_tile_max_bytes,
                    has_external_normalization=False,
                ),
                chunk_budget_bytes=chunk_budget_bytes,
            )
        mstep_block_rows = resident_chunk_memory.resolve_mstep_block_rows(
            n_recon_pixels=n_recon_windowed,
            max_block_bytes=max_adjoint_block_bytes_for_pass(
                device_memory_bytes, override=support.engine_tuning.adjoint_block_max_bytes
            ),
            row_capacity_ladder=row_ladder,
            override=support.engine_tuning.mstep_block_rows,
        )
        memory_plan = resident_chunk_memory.plan_resident_chunk_memory(
            row_capacity_ladder=row_ladder,
            image_capacity_ladder=image_ladder,
            mstep_block_rows=mstep_block_rows,
            row_bytes=resident_row_projection_bytes(
                n_score_pixels=n_windowed, n_recon_pixels=n_recon_windowed
            ),
            n_fine_trans=n_fine_trans,
            n_recon_pixels=n_recon_windowed,
            budget_bytes=chunk_budget_bytes,
            pipelined=_local_chunk_loop_pipelined(chunk_profile=support.engine_tuning.local_chunk_profile),
            projection_transient_bytes=projection_block_rows * projection_row_bytes,
            fixed_bytes=accumulator_bytes,
            max_image_rows=resident_chunk_memory.max_image_rows(tables.row_offsets),
            # An image past the largest row class runs alone in row blocks of that class
            # (resident_local_chunk.start_resident_local_chunk), as the global pass runs it (relax#49 follow-up).
            lone_row_bytes=resident_chunk_memory.lone_chunk_row_bytes(int(n_fine_trans), class_rows=tables.n_classes > 1),
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
            resident_chunk_memory.format_budget_gib(memory_plan.budget_bytes),
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
        stats_config = resident_statistics.resolve_statistics_config(
            n_shells=n_shells,
            n_fine_trans=n_fine_trans,
            n_images=n_images,
            n_coarse_rot=_posterior_bin_capacity(posterior_bin_ids.size, n_classes=tables.n_classes),
            n_scale_groups=n_scale_groups,
            # A stable-window pass's programs take the physical class; the cutoff is a runtime value below.
            current_size=wsum_current_size if stable_window_plan is None else program_current_size,
            include_unweighted_high_shell=True,
            use_exact_relion_gaussian=True,
            relion_wavg_atomic_scale_aa=relion_wavg_atomic_scale_aa,
            accumulate_scale=scale_groups_available,
            source_faithful_spectrum_norm=resolved_spectrum_norm,
            n_optics_groups=n_optics_groups,
            n_classes=tables.n_classes,
            float32_bucketed_image_sums=False,
        )
        stats = resident_statistics.make_resident_statistics(
            stats_config, max_posterior_dtype=precision_policy.score_real_dtype
        )
        # RELION's logical sizes as runtime values of a stable-window pass's programs.
        window_logical = (
            None
            if stable_window_plan is None
            else resident_stable_windows.window_logical_sizes(
                current_size=current_size,
                recon_pixels=logical_recon_pixels,
                rect_pixels=logical_rect_pixels,
                place=resident_chunk_records.PLACE_ON_DEVICE,
                mstep_max_r=resident_stable_windows.runtime_mstep_radius(mstep_max_r),
            )
        )
        image_tables = resident_image_terms.ChunkImageTables(
            shell_indices_half=jnp.asarray(shell_indices_half, dtype=jnp.int32),
            wavg_shell_indices=jnp.asarray(relion_wavg_rectangle.shell_indices, dtype=jnp.int32),
            wavg_scale_pixel_mask=jnp.asarray(scale_pixel_mask_rect_np, dtype=bool),
            translation_sqdist_ang=None,
            norm_shell_cutoff=None if window_logical is None else window_logical.current_size // 2,
        )

        exact_positions_device = jnp.asarray(relion_wavg_rectangle.exact_positions, dtype=jnp.int32)
        rect_indices_device = jnp.asarray(relion_wavg_rectangle.centered_indices, dtype=jnp.int32)
        noise_variance_for_noise_device = jnp.asarray(noise_variance_for_noise)
        shell_indices_noise_device = jnp.asarray(shell_indices_noise, dtype=jnp.int32)
        max_adjoint_block_bytes = max_adjoint_block_bytes_for_pass(
            device_memory_bytes, override=support.engine_tuning.adjoint_block_max_bytes
        )

        scale_corrections_np = (
            None
            if data.scale_corrections is None
            else np.asarray(data.scale_corrections, dtype=precision_policy.score_real_dtype)
        )
        translation_prior_centers_np = None
        if translation_prior_centers is not None:
            from relax.sampling.translation_prior import validate_translation_prior_centers

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
        pipelined = _local_chunk_loop_pipelined(chunk_profile=support.engine_tuning.local_chunk_profile)
        pending = None
        pending_alone = False
        n_lone_chunks = 0
        lone_block_sizes = set()
        for chunk in chunks:
            alone = resident_chunk_memory.chunk_runs_alone(chunk, row_ladder)
            lone_block_rows = resident_chunk_memory.lone_block_rows(chunk.row_capacity, row_ladder) if alone else None
            n_lone_chunks += int(alone)
            if alone:
                lone_block_sizes.add(lone_block_rows)
            if pending is not None and (alone or pending_alone):
                Ft_y_total, Ft_ctf_total, stats = pending(Ft_y_total, Ft_ctf_total, stats)
                pending = None
            finish = resident_local_chunk.start_resident_local_chunk(
                chunk,
                engine_tuning=support.engine_tuning,
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
                relion_projector_r_max=projection_r_max,
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
                mstep_current_size=program_volume_current_size,
                mstep_max_r=program_mstep_max_r,
                recon_volume_shape=program_recon_volume_shape,
                window_logical=window_logical,
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
                operand_route=operand_route,
                firstiter_cc=bool(kernel.firstiter_cc),
                relion_projector_capacity_texture=capacity_texture,
                class_projectors=class_projectors,
                class_scale_masks_rect=class_scale_masks_rect,
                lone_block_rows=lone_block_rows,
                row_partial=row_partial,
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
            if texture is not None and not projection_blocks.keeps_pass_texture(texture):
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
            program_recon_volume_shape=program_recon_volume_shape,
            recon_volume_shape=recon_volume_shape,
            symmetry_label=symmetry_label,
            engine_tuning=support.engine_tuning,
            n_chunks=len(chunks),
            loop_s=loop_s,
            overall_t0=overall_t0,
        )
    # RELION symmetriseReconstructions (ml_optimiser.cpp:5541-5575): x=0
    # Hermitian enforcement, then applyPointGroupSymmetry on BPref.
    Ft_y_total, Ft_ctf_total = resident_stable_windows.finalize_stable_bpref(
        Ft_y_total,
        Ft_ctf_total,
        program_recon_volume_shape=program_recon_volume_shape,
        recon_volume_shape=recon_volume_shape,
        symmetry_label=symmetry_label,
        engine_tuning=support.engine_tuning,
        logger=logger,
        label="Resident local pass-2",
    )

    finalized = resident_statistics.finalize_statistics(stats, config=stats_config, n_images=n_images)
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
    noise_stats = finalized.noise_stats()

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
    program_recon_volume_shape,
    recon_volume_shape,
    symmetry_label,
    engine_tuning,
    n_chunks,
    loop_s,
    overall_t0,
):
    """A Class3D local pass's per-class BPrefs, statistics and winners, as ``ResidentKClassPass2Output``.

    The joint statistics (normalizer, Pmax, noise and scale sums) are over classes and poses; each
    class's evidence, winner and pruned M-step mass come from its sub-segments, and its BPref is
    symmetrised on its own, as the global K-class pass does (``compute_k_class_pass2_stats_resident``).
    Winners are rows of the layout's flat order, decoded through the tables as for K=1. The BPrefs were
    accumulated in ``program_recon_volume_shape`` (the stable-window physical cube, or RELION's).
    """

    Ft_y_public, Ft_ctf_public = [], []
    for k in range(tables.n_classes):
        y, ctf = resident_stable_windows.finalize_stable_bpref(
            Ft_y_total[k],
            Ft_ctf_total[k],
            program_recon_volume_shape=program_recon_volume_shape,
            recon_volume_shape=recon_volume_shape,
            symmetry_label=symmetry_label,
            engine_tuning=engine_tuning,
            logger=logger,
            label=f"Resident local pass-2 class {k + 1}",
        )
        Ft_y_public.append(y)
        Ft_ctf_public.append(ctf)

    finalized = resident_statistics.finalize_statistics(stats, config=stats_config, n_images=n_images)
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
    noise_stats = finalized.noise_stats()
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


# The K=1 rotation posterior's used-bin count changes with every local pass, and the
# statistics programs are shaped by it: padded to this times a power of two, a refinement's
# passes share a few shapes instead of compiling the image-term program per half. A multiple
# of it gave the EMPIAR-10299 row 12 lengths and 71 image-term compiles (relax#103).
_POSTERIOR_BIN_QUANTUM = 4096


def _posterior_bin_capacity(n_used_bins: int, *, n_classes: int) -> int:
    """The accumulator length of ``n_used_bins`` used posterior bins (K>1 keeps the whole histogram).

    The smallest ``_POSTERIOR_BIN_QUANTUM * 2**k`` that holds them: at most twice the used bins.
    """

    if n_classes != 1:
        return int(n_used_bins)
    blocks = max(1, -(-int(n_used_bins) // _POSTERIOR_BIN_QUANTUM))
    return (1 << (blocks - 1).bit_length()) * _POSTERIOR_BIN_QUANTUM


def _expand_posterior_bins(sums, posterior_bin_ids, n_posterior_bins: int) -> np.ndarray:
    """The rotation posterior over the layout's whole histogram, from its used bins (host float64).

    ``sums`` may be padded past the used bins (:func:`_posterior_bin_capacity`); the padding holds no mass.
    """

    used = np.asarray(posterior_bin_ids, dtype=np.int64)
    dense = np.zeros(int(n_posterior_bins), dtype=np.float64)
    dense[used] = np.asarray(jax.device_get(sums), dtype=np.float64)[: used.size]
    return dense
