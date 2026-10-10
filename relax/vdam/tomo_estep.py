"""InitialModel (VDAM) E-step of subtomogram particles (RELION 5 2D stacks).

RELION's gradient E-step of a tilt-series particle is the subtomogram refinement E-step
(``expectationOneParticle`` over its tilt images, one pose and one posterior per particle) with
VDAM's three differences (:mod:`relax.vdam.adaptive_estep`): every tilt image backprojects its
residual ``shift(X_i) - CTF_i P_i V`` (BP.cuh ``backproject3D_SGD``), into the BPref slot
``class + K * (part_id % 2)`` of its particle (acc_ml_optimiser_impl.h:3391-3395), and the coarse
pass keeps at most ``max_significants`` samples. The scoring is
:func:`relax.refinement.tomo_scoring.score_tomo_half`, the subtomogram Refine3D/Class3D pass; this
module turns its output into VDAM's accumulators and E-step metadata, as the single-particle
route does (:func:`relax.vdam.adaptive_estep.run_adaptive_initial_model_estep`).
"""

from __future__ import annotations

import logging

import numpy as np

from relax import sampling
from relax.refinement.scoring_policy import RELION_ADAPTIVE_FRACTION
from relax.vdam.adaptive_estep import (
    direction_posterior_stats,
    recovar_order_prior,
    relion_order_of_recovar_rotations,
    resolve_sparse_pass1_current_size,
    sparse_pass2_estep_meta,
)
from relax.vdam.estep_common import (
    TOMO_META_PARTICLE_FIELDS,
    InitialModelEstepResult,
    add_accumulator_weight_meta,
    arrays_to_accumulators,
    empty_accumulator,
)
from relax.vdam.state import InitialModelState

logger = logging.getLogger(__name__)

__all__ = ["run_tomo_initial_model_estep", "tomo_initial_model_sampling"]


def tomo_initial_model_sampling(
    state: InitialModelState, sampling_plan, *, particle_diameter_ang: float, pass1_healpix_order: int | None = None
):
    """The iteration's :class:`relax.refinement.tomo_half.TomoSampling` from VDAM's sampling plan.

    The coarse size is RELION's ``image_coarse_size`` for the pass-1 order, as for single particles
    (:func:`relax.vdam.adaptive_estep.resolve_sparse_pass1_current_size`); the fine size is the
    current size. ``pass1_healpix_order`` is the order before this iteration's sampling update:
    RELION sets the sizes (expectationSetup step A) before it updates the sampling (step D).
    """

    from relax.refinement.tomo_half import TomoSampling

    fine_size = int(state.effective_current_size)
    order = int(sampling_plan.healpix_order)
    coarse_size = (
        fine_size
        if int(sampling_plan.oversampling) == 0
        else resolve_sparse_pass1_current_size(
            state,
            None if int(state.current_size) <= 0 else int(state.current_size),
            float(particle_diameter_ang),
            order if pass1_healpix_order is None else int(pass1_healpix_order),
        )
    )
    return TomoSampling(
        healpix_order=order,
        oversampling_order=int(sampling_plan.oversampling),
        offset_range_angst=float(sampling_plan.offset_range_angstrom),
        offset_step_angst=float(sampling_plan.offset_step_angstrom),
        random_perturbation=float(sampling_plan.random_perturbation),
        coarse_size=int(state.box_size) if coarse_size is None else int(coarse_size),
        fine_size=fine_size,
    )


def run_tomo_initial_model_estep(
    tomo_dataset,
    state: InitialModelState,
    *,
    sampling_plan,
    particle_ids,
    halfset_ids,
    previous_offsets_px,
    noise_variance,
    relion_projector_half_by_class,
    relion_projector_r_max: int,
    class_rotation_log_prior,
    max_significants: int,
    sigma_offset_angstrom: float,
    particle_diameter_ang: float,
    padding_factor: int,
    optics_group_ids=None,
    pass1_healpix_order: int | None = None,
) -> InitialModelEstepResult:
    """One VDAM E-step over the subset's particles, both pseudo-halfsets in one pass.

    ``particle_ids`` are particle-STAR rows (the dataset's units) and ``halfset_ids`` their
    pseudo-halfsets; ``previous_offsets_px`` ``[P, 3]`` the subset's previous 3D offsets (pixels);
    ``class_rotation_log_prior`` ``[K, R]`` VDAM's joint class/direction prior in RELION's
    direction-major order. With several optics groups ``noise_variance`` is ``[G, N^2]`` and
    ``optics_group_ids`` gives every particle-STAR row's group. The metadata carries the particles' new offsets
    (``tomo_offsets_px``: RELION's rounded old offset plus the winning shift).
    """

    from relax.classification.k_class import _class_segmented_em_result, single_class_pass2_em_result
    from relax.reconstruction.half_volume_mstep import relion_backprojector_volume_shape
    from relax.refinement import tomo_particles
    from relax.refinement.tomo_half import tomo_translation_grids
    from relax.refinement.tomo_scoring import score_tomo_half

    particle_ids = np.asarray(particle_ids, dtype=np.int64).reshape(-1)
    if particle_ids.size == 0:
        empty = [empty_accumulator(state, k, h) for h in (0, 1) for k in range(state.K)]
        return InitialModelEstepResult(accumulators=empty, meta={"pass2_engine": "tomo"})
    if not state.pseudo_halfsets:
        raise NotImplementedError("subtomogram InitialModel backprojects into RELION's two pseudo-halfsets")
    group_ids = np.asarray(halfset_ids, dtype=np.int32).reshape(-1)
    if group_ids.shape != particle_ids.shape or np.any((group_ids != 0) & (group_ids != 1)):
        raise ValueError("pseudo-halfset ids must give each selected particle 0 or 1")
    half = tomo_dataset.subset(particle_ids)
    tomo_sampling = tomo_initial_model_sampling(
        state, sampling_plan, particle_diameter_ang=particle_diameter_ang, pass1_healpix_order=pass1_healpix_order
    )
    order = int(tomo_sampling.healpix_order)
    n_coarse_rot = int(sampling.rotation_grid_size(order))
    relion_of_recovar = relion_order_of_recovar_rotations(order)
    prior = np.asarray(recovar_order_prior(np.asarray(class_rotation_log_prior), relion_of_recovar), dtype=np.float32)
    if prior.shape != (state.K, n_coarse_rot):
        raise ValueError(f"the class/direction prior must be [K, R] = {(state.K, n_coarse_rot)}, got {prior.shape}")
    projector_half = np.asarray(relion_projector_half_by_class)
    old = np.asarray(previous_offsets_px, dtype=np.float64).reshape(particle_ids.size, 3)
    # The resident pass reads only the projector halves (a NaN stand-in shows any other read).
    volume_stand_in = np.full((state.K, 1) if state.K > 1 else (1,), np.nan, dtype=np.complex64)
    scored = score_tomo_half(
        half,
        volume=volume_stand_in,
        noise_variance=noise_variance,
        relion_projector_half=projector_half if state.K > 1 else projector_half[0],
        relion_projector_r_max=int(relion_projector_r_max),
        sampling=tomo_sampling,
        rotation_log_prior=prior if state.K > 1 else prior[0],
        old_offsets_px=old,
        sigma_offset_angst=float(sigma_offset_angstrom),
        adaptive_fraction=RELION_ADAPTIVE_FRACTION,
        max_significants=int(max_significants),
        unit_groups=(
            np.zeros(particle_ids.size, dtype=np.int32)
            if optics_group_ids is None
            else np.asarray(optics_group_ids, dtype=np.int32)[particle_ids]
        ),
        padding_factor=int(padding_factor),
        reconstruction_group_ids=group_ids,
        reconstruction_group_count=2,
        mstep_subtract_ctf_projection=True,
    )
    counts = np.asarray(scored.significant_counts)
    logger.info(
        "Subtomogram VDAM E-step: %d particles, HEALPix %d, sizes %d/%d, max_significants %d, significant samples "
        "mean %.1f max %d",
        particle_ids.size,
        order,
        tomo_sampling.coarse_size,
        tomo_sampling.fine_size,
        int(max_significants),
        float(np.mean(counts)),
        int(np.max(counts)),
    )
    _, _, fine_px, _ = tomo_translation_grids(tomo_sampling, float(half.voxel_size))
    mstep_shape = relion_backprojector_volume_shape(
        half.volume_shape, int(padding_factor), current_size=int(tomo_sampling.fine_size)
    )
    if state.K > 1:
        result = _class_segmented_em_result(
            scored.pass2,
            n_classes=state.K,
            class_posterior_sums_from_noise=True,
            return_profile=False,
            host_accumulators=True,
            mstep_full_half_axis=0,
            mstep_accumulator_shape=mstep_shape,
        )
    else:
        result = single_class_pass2_em_result(
            scored.pass2,
            n_fine_trans=int(fine_px.shape[0]),
            accumulate_noise=True,
            return_best_pose_details=True,
            mstep_full_half_axis=0,
            mstep_accumulator_shape=mstep_shape,
        )
    # The tilt pass sums its rotation posteriors over the coarse grid (RECOVAR order).
    result = direction_posterior_stats(
        result,
        n_coarse_rot=n_coarse_rot,
        rot_parent_map=np.arange(n_coarse_rot, dtype=np.int64),
        n_psi=int(sampling.rotation_grid_n_in_planes(order)),
    )
    accumulators = arrays_to_accumulators(
        result.Ft_y,
        result.Ft_ctf,
        state,
        half_count=2,
        padding_factor=int(padding_factor),
    )
    # vdam_m_step reads the list halfset-major (accumulators[k], accumulators[K + k]).
    accumulators = sorted(accumulators, key=lambda accum: (accum.halfset_idx, accum.class_idx))
    meta = sparse_pass2_estep_meta(result, particle_ids, particle_fields=TOMO_META_PARTICLE_FIELDS)
    # The particles' offsets are 3D; RELION writes the rounded old offset plus the winning shift.
    pose = np.asarray(result.pose_assignments, dtype=np.int64)
    meta["tomo_offsets_px"] = (
        tomo_particles.relion_gpu_old_offsets(old) + np.asarray(fine_px, dtype=np.float64)[pose % int(fine_px.shape[0])]
    )
    meta.update(offset_dims=3, significant_counts=counts.astype(np.int32))  # nsig as Refine3D writes it
    add_accumulator_weight_meta(meta, accumulators, state.K)
    meta["pass2_engine"] = "tomo"
    meta["halfset_ids"] = (0, 1)
    meta["joint_halfset_particle_stream"] = True
    return InitialModelEstepResult(accumulators=accumulators, meta=meta)
