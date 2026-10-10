"""The E-step and M-step of a subtomogram half (:class:`relax.refinement.tomo_half.TomoHalf`): the coarse pass
per particle (:mod:`relax.scoring.tomo_coarse`) and the resident fine pass and M-step over tilt units
(``resident_pass2._resident_pass2(tilt=...)``), and the numbered iteration's call of them."""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

from relax.refinement import tomo_particles
from relax.refinement.half_inputs import HalfScoringData
from relax.refinement.scoring_policy import RELION_ADAPTIVE_FRACTION
from relax.refinement.tomo_half import TomoHalf, TomoSampling, tomo_local_rotations, tomo_translation_grids
from relax.relion.geometry import (
    PROJECTION_PADDING_FACTOR,
    RECONSTRUCTION_PADDING_FACTOR,
)

logger = logging.getLogger(__name__)


def tilt_pass_inputs(
    half: TomoHalf,
    *,
    fine_px,
    fine_parent,
    unit_coarse_prior,
    old_offsets_px,
    pixel_size: float,
    fine_source_eulers,
    fine_rotations,
    unit_groups,
):
    """The resident pass's :class:`TiltPassInputs` for one half and one sampling."""

    from relax.fine_pass.resident_tilts import TiltPassInputs

    image_particle = half.image_particle()
    rounded_old = tomo_particles.relion_gpu_old_offsets(old_offsets_px)
    # sigma2_offset sums: pixel_size^2 |rounded old + trial shift - prior|^2, prior zero in auto-refine
    # (acc_ml_optimiser_impl.h:2700-2736, :2845).
    shifted = rounded_old[:, None, :] + np.asarray(fine_px, dtype=np.float64)[None, :, :]
    return TiltPassInputs(
        unit_image_offsets=half.unit_image_offsets,
        image_left=half.image_left,
        image_angles=tomo_particles.tilt_translation_angles(
            fine_px, rounded_old, half.image_projections, image_particle, half.grid_size
        ),
        image_noise_scale=tomo_particles.image_noise_scale(image_particle, half.n_units).astype(np.float32),
        unit_translation_prior=np.asarray(np.asarray(unit_coarse_prior)[:, fine_parent], dtype=np.float32),
        unit_translation_sqdist_ang=float(pixel_size) ** 2 * np.sum(shifted * shifted, axis=-1),
        unit_optics_groups=np.asarray(unit_groups, dtype=np.int32),
        fine_source_eulers=fine_source_eulers,
        fine_rotations=fine_rotations,
        slot_capacity=int(np.max(np.diff(half.unit_image_offsets))) if half.n_units else 1,
    )


@dataclasses.dataclass(frozen=True)
class TomoScoreResult:
    """One tomo half's E- and M-step, per particle (the units of the half)."""

    # K=1: SparsePass2Output of compute_tilt_pass2_stats_resident (Ft, per-particle poses and stats);
    # K>1: ResidentKClassPass2Output of compute_k_class_pass2_stats_resident.
    pass2: object
    coarse_hard_assignment: np.ndarray  # int32 [P], coarse rotation * T_coarse + coarse translation
    significant_counts: np.ndarray  # int32 [P], coarse significant samples


def score_tomo_half(
    half: TomoHalf,
    *,
    volume,
    noise_variance,
    relion_projector_half,
    relion_projector_r_max: int,
    sampling: TomoSampling,
    rotation_log_prior,
    old_offsets_px,
    sigma_offset_angst: float,
    adaptive_fraction: float,
    max_significants,
    unit_groups,
    padding_factor: int,
    scale_corrections=None,
    group_ids=None,
    scale_correction_group_count=None,
    scale_correction_data_vs_prior=None,
    reconstruction_current_size=None,
    local_rotations=None,
    symmetry: str = "C1",
    unit_seed_classes=None,
    reconstruction_group_ids=None,
    reconstruction_group_count=None,
    mstep_subtract_ctf_projection: bool = False,
    normalized_cc: bool = False,
    coarse_cc_winners=None,
) -> TomoScoreResult:
    """RELION's adaptive two-pass E-step and M-step of subtomogram particles (global search).

    Pass 1 scores every particle's tilt images on the coarse grid and cuts the particle's summed
    posterior (:func:`relax.scoring.tomo_coarse.particle_coarse_supports`); pass 2 scores the
    significant samples' oversampled children and backprojects every tilt image
    (``resident_pass2.compute_tilt_pass2_stats_resident``). ``old_offsets_px`` are the particles'
    previous 3D offsets (unrounded, pixels); ``noise_variance`` is one spectrum or ``[G, N^2]`` rows
    with ``unit_groups`` the dense optics group of each particle. The flags are the production K=1
    ones (the one-iteration RELION-pinned replay, em_work/cryoet_s42_20260925/tomo_replay_it1.py).
    The coarse grid is RELION's HEALPix grid at ``sampling.healpix_order`` in the loop's ("recovar")
    rotation order, so ``rotation_log_prior`` and the returned rotation sums share it.

    A local search passes ``local_rotations`` (:func:`tomo_local_rotations`): the coarse grid is then
    the union of the particles' local rotations at ``sampling.healpix_order`` and each particle is
    scored over its own with its own prior (``rotation_log_prior`` must be None); the returned rotation
    sums are over that union.

    Class3D (K>1): ``volume`` and ``relion_projector_half`` stack the K class references and
    ``rotation_log_prior`` is ``[K, R]``, each class's direction prior with its ``log pdf_class``. Pass 1
    cuts each particle's weights over every class jointly and pass 2 scores all classes in the
    particle's one posterior segment, each class backprojecting into its own BPref
    (``resident_pass2.compute_k_class_pass2_stats_resident``). A global search only. ``unit_seed_classes``
    ``[P]`` is RELION's first iteration from one reference: each particle is scored against that class
    only (``k_class_inputs.seed_iteration_supports``; its other classes have no candidates).

    InitialModel (VDAM, ``relax.vdam.tomo_estep``) passes ``reconstruction_group_ids`` ``[P]``, each
    particle's pseudo-halfset (RELION's ``part_id % 2``: every tilt image of a particle backprojects into
    BPref slot ``class + K * group``, acc_ml_optimiser_impl.h:3391-3395), with ``reconstruction_group_count``
    2, and ``mstep_subtract_ctf_projection``: each tilt image backprojects its residual
    ``shift(X_i) - CTF_i P_i V`` (BP.cuh ``backproject3D_SGD``).

    ``normalized_cc`` is RELION's ``--firstiter_cc`` iteration (a global search): every image's
    normalized CC summed over the particle's tilt images, the coarse winner alone
    (:func:`tilt_coarse_cc_winners`; ``coarse_cc_winners`` ``[P]`` gives them when the caller has them) and its
    children scored the same way in pass 2, the best cell taking all (``resident_tilts.tilt_cc_scores``); no
    priors. With K class references and no seed classes every class is scored and each particle goes to the
    class of its best coarse sample (:func:`_score_class_references_cc`).
    """

    import jax.numpy as jnp
    from recovar.reconstruction import noise as recon_noise

    from relax import sampling as relax_sampling
    from relax.classification.k_class_inputs import seed_iteration_first_class, seed_iteration_supports
    from relax.fine_pass.resident_pass2 import (
        compute_k_class_pass2_stats_resident,
        compute_tilt_pass2_stats_resident,
    )
    from relax.scoring import tomo_coarse

    # RELION's float projector, as the SPA passes read it: the refinement builds complex128, and a complex128
    # projector never reaches the float32 texture, so the coarse pass projected in double where RELION interpolates
    # its float texture (et13_k2conf left RELION's trajectory at it002, 2026-10-03).
    relion_projector_half = np.asarray(relion_projector_half, dtype=np.complex64)
    n_classes = int(relion_projector_half.shape[0]) if relion_projector_half.ndim == 4 else 1
    if normalized_cc and n_classes > 1:
        if unit_seed_classes is None:
            # One reference per class: RELION restricts the CC iteration to the first reference only when it
            # generates seeds (do_generate_seeds, acc_ml_optimiser_impl.h:3696-3707; ml_model.cpp:1008-1010).
            return _score_class_references_cc(
                half, volume=volume, noise_variance=noise_variance, relion_projector_half=relion_projector_half,
                relion_projector_r_max=relion_projector_r_max, sampling=sampling, rotation_log_prior=rotation_log_prior,
                old_offsets_px=old_offsets_px, sigma_offset_angst=sigma_offset_angst, adaptive_fraction=adaptive_fraction,
                max_significants=max_significants, unit_groups=unit_groups, padding_factor=padding_factor,
                scale_corrections=scale_corrections, group_ids=group_ids,
                scale_correction_group_count=scale_correction_group_count,
                scale_correction_data_vs_prior=scale_correction_data_vs_prior,
                reconstruction_current_size=reconstruction_current_size, symmetry=symmetry,
            )
        if np.any(np.asarray(unit_seed_classes) != 0):
            # One reference for K classes: the CC iteration is K=1 against it (ml_optimiser.cpp:4389-4402).
            raise ValueError("a seeded K-class --firstiter_cc iteration scores every particle against class 0")
        # The loop copies the first class's model to every class and seeds the classes in the next iteration.
        first = score_tomo_half(
            half,
            volume=np.asarray(volume)[0],
            noise_variance=noise_variance,
            relion_projector_half=relion_projector_half[0],
            relion_projector_r_max=relion_projector_r_max,
            sampling=sampling,
            rotation_log_prior=np.asarray(rotation_log_prior)[0],
            old_offsets_px=old_offsets_px,
            sigma_offset_angst=sigma_offset_angst,
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants,
            unit_groups=unit_groups,
            padding_factor=padding_factor,
            scale_corrections=scale_corrections,
            group_ids=group_ids,
            scale_correction_group_count=scale_correction_group_count,
            scale_correction_data_vs_prior=scale_correction_data_vs_prior,
            reconstruction_current_size=reconstruction_current_size,
            symmetry=symmetry,
            normalized_cc=True,
        )
        return dataclasses.replace(first, pass2=_first_class_k_class_output(first.pass2, n_classes, sampling, half.voxel_size))
    if n_classes > 1 and (local_rotations is not None or np.ndim(rotation_log_prior) != 2):
        raise ValueError("a K-class tomo pass is a global search with one rotation log prior per class, [K, R]")
    pixel = float(half.voxel_size)
    size = int(half.grid_size)
    unit_groups = np.asarray(unit_groups, dtype=np.int32)
    image_groups = np.repeat(unit_groups, np.diff(half.unit_image_offsets))
    old_offsets_px = np.asarray(old_offsets_px, dtype=np.float64).reshape(half.n_units, 3)
    coarse_angst, coarse_px, fine_px, fine_parent = tomo_translation_grids(sampling, pixel)
    if local_rotations is not None and rotation_log_prior is not None:
        raise ValueError("a local search's priors are the particles' own")
    coarse_ids = (
        np.arange(int(relax_sampling.rotation_grid_size(sampling.healpix_order, symmetry)))
        if local_rotations is None
        else local_rotations.coarse_rotation_ids
    )
    n_rot = int(coarse_ids.size)
    coarse_eulers_deg = relax_sampling.rotation_indices_to_relion_eulers(
        coarse_ids, sampling.healpix_order, rotation_index_order="recovar", symmetry=symmetry
    )
    fine_rot, rot_parent, fine_mstep, fine_eulers = relax_sampling.get_oversampled_rotation_grid_from_samples(
        coarse_ids,
        sampling.healpix_order,
        oversampling_order=sampling.oversampling_order,
        random_perturbation=sampling.random_perturbation,
        return_mstep_rotations=True,
        return_source_eulers=True,
        rotation_index_order="recovar",
        symmetry=symmetry,
        dtype=np.float32,
    )
    unit_coarse_prior = tomo_particles.relion_offset_log_prior_3d(
        coarse_angst, old_offsets_px, pixel_size=pixel, sigma_offset_angst=sigma_offset_angst
    )
    noise = jnp.asarray(noise_variance)
    noise_half = recon_noise.to_batched_half_pixel_noise(noise, (size, size))
    if noise_half.ndim == 2 and noise_half.shape[0] == 1:
        noise_half = noise_half[0]
    layout = tomo_coarse.coarse_score_layout(
        (size, size), sampling.coarse_size, half_spectrum_scoring=True, square_window=False
    )
    image_scale = None if scale_corrections is None else np.repeat(
        np.asarray(scale_corrections, dtype=np.float32), np.diff(half.unit_image_offsets)
    )
    # A seed iteration scores every particle against its random class alone: pass 1 against the first copy of
    # the one reference (k_class_inputs.seed_iteration_first_class), each support then in the particle's class.
    coarse_halves, coarse_prior = relion_projector_half, rotation_log_prior
    if unit_seed_classes is not None:
        if n_classes == 1:
            raise ValueError("a seed iteration is a K-class pass")
        (coarse_halves,), (coarse_prior,) = seed_iteration_first_class(relion_projector_half, rotation_log_prior)
    if normalized_cc:
        if n_classes != 1 or local_rotations is not None or symmetry != "C1":
            raise NotImplementedError("the subtomogram --firstiter_cc iteration is a K=1 global search in C1")
        winners = coarse_cc_winners
        if winners is None:
            winners, _best_cc = tilt_coarse_cc_winners(
                half, sampling=sampling, old_offsets_px=old_offsets_px, relion_projector_half=relion_projector_half,
                relion_projector_r_max=relion_projector_r_max, padding_factor=padding_factor,
                scale_corrections=scale_corrections,
            )
        supports = [np.asarray([cell], dtype=np.int32) for cell in winners]
    else:
        supports, _coarse_pmax = tomo_coarse.particle_coarse_supports(
            half.images,
            unit_image_offsets=half.unit_image_offsets,
            image_projections=half.image_projections,
            image_left=half.image_left,
            unit_old_offsets_px=old_offsets_px,
            coarse_eulers_deg=coarse_eulers_deg,
            random_perturbation=sampling.random_perturbation,
            angular_sampling_deg=relax_sampling.relion_angular_sampling_deg(sampling.healpix_order),
            coarse_translations_px=coarse_px,
            projector_half=(
                jnp.asarray(coarse_halves)
                if np.ndim(coarse_halves) == 3
                else tuple(jnp.asarray(class_half) for class_half in coarse_halves)
            ),
            layout=layout,
            noise_variance_half=noise_half,
            rotation_log_prior=coarse_prior,
            unit_translation_log_prior=unit_coarse_prior,
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants,
            model_max_r=int(relion_projector_r_max),
            padding_factor=int(padding_factor),
            box_size=size,
            optics_group_ids=image_groups,
            scale_corrections=image_scale,
            **(
                {}
                if local_rotations is None
                else {
                    "unit_rotation_ids": local_rotations.unit_rotation_ids,
                    "unit_rotation_log_priors": local_rotations.unit_rotation_log_priors,
                }
            ),
        )
    if unit_seed_classes is not None:
        supports = seed_iteration_supports(supports, unit_seed_classes, n_classes)
    tilt = tilt_pass_inputs(
        half,
        fine_px=fine_px,
        fine_parent=fine_parent,
        unit_coarse_prior=unit_coarse_prior,
        old_offsets_px=old_offsets_px,
        pixel_size=pixel,
        fine_source_eulers=fine_eulers,
        fine_rotations=fine_rot,
        unit_groups=unit_groups,
    )
    image_group_ids = None if group_ids is None else np.repeat(
        np.asarray(group_ids, dtype=np.int32), np.diff(half.unit_image_offsets)
    )
    options = dict(
        # RELION's asymmetric-unit grid and its symmetrised BPref (symmetriseReconstructions).
        symmetry_label=symmetry,
        oversampling_order=sampling.oversampling_order,
        current_size=sampling.fine_size,
        reconstruction_current_size=(
            sampling.fine_size if reconstruction_current_size is None else int(reconstruction_current_size)
        ),
        translation_step=None,
        score_with_masked_images=True,
        return_stats=True,
        translation_log_prior=None,
        accumulate_noise=True,
        half_spectrum_scoring=True,
        projection_padding_factor=int(padding_factor),
        reconstruction_padding_factor=int(padding_factor),
        # RELION neither normalises nor pre-shifts a tilt image (acc :429-476), but backprojects it with its
        # group-scaled CTF (ctf * scale, acc_ml_optimiser_impl.h:4370-4404): the image correction is the scale.
        image_corrections=image_scale,
        scale_corrections=image_scale,
        image_pre_shifts=None,
        use_float64_scoring=False,
        do_gridding_correction=True,
        random_perturbation=sampling.random_perturbation,
        group_ids=image_group_ids,
        scale_correction_group_count=scale_correction_group_count,
        scale_correction_data_vs_prior=scale_correction_data_vs_prior,
        normalization_score_mode="normalized_cc" if normalized_cc else "gaussian",
        relion_firstiter_score_mode="normalized_cc" if normalized_cc else "gaussian",
        relion_firstiter_winner_take_all=bool(normalized_cc),
        return_score_log_z=True,
        return_source_eulers=True,
        fine_source_eulers_override=fine_eulers,
        fine_rotations_override=fine_rot,
        fine_mstep_rotations_override=fine_mstep,
        fine_rotation_parent_override=rot_parent,
        fine_translations_override=fine_px,
        fine_translation_parent_override=fine_parent,
        relion_x_half_mstep=True,
        relion_fine_mstep_prune=True,
        relion_exact_fine_gaussian=True,
        relion_fine_diff2_fused_ffi=True,
        relion_f32_fine_posterior=True,
        relion_projector_half=relion_projector_half,
        relion_projector_r_max=int(relion_projector_r_max),
        adaptive_fraction=adaptive_fraction,
        include_unweighted_norm_high_shell=True,
        preserve_bpref_particle_order=True,
        source_faithful_spectrum_norm=True,
        optics_group_ids=image_groups if np.asarray(noise_variance).ndim == 2 else None,
        tilt=tilt,
        # The candidate tables are the particles', so a reconstruction group is a particle's.
        reconstruction_group_ids=None if reconstruction_group_ids is None else np.asarray(reconstruction_group_ids, dtype=np.int32),
        reconstruction_group_count=None if reconstruction_group_count is None else int(reconstruction_group_count),
        mstep_subtract_ctf_projection=bool(mstep_subtract_ctf_projection),
        **(
            {}
            if local_rotations is None
            else {
                "coarse_rotation_ids": coarse_ids,
                "unit_rotation_log_prior": local_rotations.support_priors(supports, coarse_px.shape[0]),
            }
        ),
    )
    if n_classes == 1:
        pass2 = compute_tilt_pass2_stats_resident(
            experiment_dataset=half.images,
            volume=volume,
            noise_variance=noise,
            translations=coarse_px,
            significant_sample_indices=supports,
            nside_level=sampling.healpix_order,
            disc_type="linear_interp",
            rotation_log_prior=rotation_log_prior,
            relion_exact_fine_normalized_cc=True,
            **options,
        )
        hard = np.asarray(pass2.hard_assignment, dtype=np.int64)
        significant_counts = np.asarray([s.size for s in supports], dtype=np.int32)
    else:
        pass2 = compute_k_class_pass2_stats_resident(
            half.images,
            volume,
            noise,
            coarse_px,
            supports,
            sampling.healpix_order,
            "linear_interp",
            rotation_log_priors_by_class=[np.asarray(prior, dtype=np.float32) for prior in rotation_log_prior],
            **options,
        )
        # The particle's class is its best-scoring class (k_class_results._assemble_result's argmax).
        winner = np.argmax(np.asarray(pass2.class_best_log_score_per_image, dtype=np.float64), axis=0)
        units = np.arange(half.n_units)
        hard = np.asarray(pass2.per_class_hard_assignments, dtype=np.int64)[winner, units]
        significant_counts = np.sum([[s.size for s in class_supports] for class_supports in supports], axis=0)
    n_fine_trans = int(fine_px.shape[0])
    coarse_hard = rot_parent[hard // n_fine_trans] * int(coarse_px.shape[0]) + fine_parent[hard % n_fine_trans]
    return TomoScoreResult(
        pass2=pass2,
        coarse_hard_assignment=coarse_hard.astype(np.int32),
        significant_counts=np.asarray(significant_counts, dtype=np.int32),
    )


def tilt_coarse_cc_winners(
    half: TomoHalf, *, sampling: TomoSampling, old_offsets_px, relion_projector_half, relion_projector_r_max: int,
    padding_factor: int, scale_corrections=None,
):
    """Each particle's coarse ``--firstiter_cc`` winner against one reference on the global C1 grid: the cell
    ``rotation * T + translation`` and its normalized CC (:func:`relax.scoring.tomo_coarse.particle_coarse_cc_winners`).
    ``scale_corrections`` are per particle."""

    from relax import sampling as relax_sampling
    from relax.fine_pass.bucket_arrays import relion_parent_execution_key
    from relax.scoring import tomo_coarse

    coarse_ids = np.arange(int(relax_sampling.rotation_grid_size(sampling.healpix_order, "C1")))
    _, coarse_px, _, _ = tomo_translation_grids(sampling, float(half.voxel_size))
    return tomo_coarse.particle_coarse_cc_winners(
        half.images,
        unit_image_offsets=half.unit_image_offsets,
        image_projections=half.image_projections,
        image_left=half.image_left,
        unit_old_offsets_px=np.asarray(old_offsets_px, dtype=np.float64).reshape(half.n_units, 3),
        coarse_eulers_deg=relax_sampling.rotation_indices_to_relion_eulers(
            coarse_ids, sampling.healpix_order, rotation_index_order="recovar", symmetry="C1"
        ),
        relion_order=relion_parent_execution_key(
            np.asarray(coarse_ids, dtype=np.int64), n_coarse_rot=int(coarse_ids.size), nside_level=sampling.healpix_order
        ),
        random_perturbation=sampling.random_perturbation,
        angular_sampling_deg=relax_sampling.relion_angular_sampling_deg(sampling.healpix_order),
        coarse_translations_px=coarse_px,
        relion_projector_half=relion_projector_half,
        relion_projector_r_max=int(relion_projector_r_max),
        padding_factor=int(padding_factor),
        coarse_size=int(sampling.coarse_size),
        box_size=int(half.grid_size),
        scale_corrections=None if scale_corrections is None else np.repeat(
            np.asarray(scale_corrections, dtype=np.float32), np.diff(half.unit_image_offsets)
        ),
    )


def classes_of_best_coarse_cc(best_cc_by_class) -> np.ndarray:
    """Each particle's class in a ``--firstiter_cc`` iteration from K references: the class of its best coarse
    sample, the first class on an exact tie.

    RELION takes the device arg-min of the float32 ``diff2 = -CC`` over one class-major weight array
    (acc_ml_optimiser_impl.h:1384-1392 and 2012-2026; cuda_utils_cub.cuh:66-84), so equal scores go to the
    lowest class. ``best_cc_by_class`` is ``[K, P]``; the comparison is in float32.
    """

    return np.argmax(np.asarray(best_cc_by_class, dtype=np.float32), axis=0)


def _score_class_references_cc(half, *, volume, relion_projector_half, rotation_log_prior, old_offsets_px, unit_groups,
                               scale_corrections, group_ids, sampling, scale_correction_data_vs_prior,
                               relion_projector_r_max, padding_factor, **common):
    """RELION's ``--firstiter_cc`` iteration from one reference per class (relax#75).

    Every class is scored: each particle's coarse sample is the best over the classes
    (:func:`classes_of_best_coarse_cc`), and its children are scored against that class alone, the best cell
    taking the whole weight. So each class runs the K=1 pass over its own particles with the coarse winners
    already found, as the single-particle engine does (``k_class._run_sparse_firstiter_global_winner_subset_pass2``).
    """

    n_classes, n_units = int(np.shape(relion_projector_half)[0]), int(half.n_units)
    old = np.asarray(old_offsets_px, dtype=np.float64).reshape(n_units, 3)
    groups = np.asarray(unit_groups)
    scales = None if scale_corrections is None else np.asarray(scale_corrections)
    dvp = None if scale_correction_data_vs_prior is None else np.asarray(scale_correction_data_vs_prior)
    coarse = [
        tilt_coarse_cc_winners(
            half, sampling=sampling, old_offsets_px=old, relion_projector_half=np.asarray(relion_projector_half)[k],
            relion_projector_r_max=relion_projector_r_max, padding_factor=padding_factor, scale_corrections=scales,
        )
        for k in range(n_classes)
    ]
    unit_class = classes_of_best_coarse_cc(np.stack([np.asarray(cc) for _, cc in coarse]))
    members = [np.flatnonzero(unit_class == k) for k in range(n_classes)]
    logger.info(
        "Subtomogram --firstiter_cc from %d class references: particles per class %s",
        n_classes, [int(m.size) for m in members],
    )
    results = [
        None
        if members[k].size == 0
        else score_tomo_half(
            half.subset(members[k]),
            volume=np.asarray(volume)[k],
            relion_projector_half=np.asarray(relion_projector_half)[k],
            relion_projector_r_max=relion_projector_r_max,
            padding_factor=padding_factor,
            sampling=sampling,
            rotation_log_prior=np.asarray(rotation_log_prior)[k],
            old_offsets_px=old[members[k]],
            unit_groups=groups[members[k]],
            scale_corrections=None if scales is None else scales[members[k]],
            group_ids=None if group_ids is None else np.asarray(group_ids)[members[k]],
            scale_correction_data_vs_prior=dvp if dvp is None or dvp.ndim == 1 else dvp[k],
            normalized_cc=True,
            coarse_cc_winners=np.asarray(coarse[k][0])[members[k]],
            **common,
        )
        for k in range(n_classes)
    ]
    return winner_class_tomo_result(
        results, members, n_units, sampling, float(half.voxel_size),
        scale_group_count=common.get("scale_correction_group_count"),
    )


def winner_class_tomo_result(results, members, n_units: int, sampling: TomoSampling, pixel_size: float, *,
                             scale_group_count=None) -> TomoScoreResult:
    """K=1 tilt passes over disjoint particle sets, one per class, as the K-class pass (``ResidentKClassPass2Output``).

    ``results[k]`` is class k's :class:`TomoScoreResult` over the particles ``members[k]`` (None for a class without
    particles). A particle has candidates in its own class only: the other classes' evidence and best scores are
    ``-inf``, their poses ``-1`` and zero, as :func:`_first_class_k_class_output` gives a single-class iteration.
    """

    from relax.classification.k_class import sparse_pose_ids_to_fine_grid
    from relax.classification.k_class_results import _expand_subset_noise_stats, _sum_noise_stats
    from relax.fine_pass.resident_pass2 import ResidentKClassPass2Output
    from relax.types import make_relion_stats

    n_classes = len(results)
    present = [(k, r) for k, r in enumerate(results) if r is not None]
    first = present[0][1].pass2
    _, _, fine_px, _ = tomo_translation_grids(sampling, pixel_size)
    unit_class = np.zeros(n_units, dtype=np.int64)
    for k, _ in present:
        unit_class[members[k]] = k

    def per_unit(get, fill, dtype):
        out = np.full((n_classes, n_units), fill, dtype=dtype)
        for k, r in present:
            out[k, members[k]] = np.asarray(get(r.pass2), dtype=dtype)
        return out

    def per_class(get, like):
        like = np.asarray(like)
        out = [np.zeros((n_units,) + like.shape[1:], dtype=like.dtype) for _ in range(n_classes)]
        for k, r in present:
            out[k][members[k]] = np.asarray(get(r.pass2))
        return tuple(out)

    log_evidence = per_unit(lambda p: p.relion_stats.log_evidence_per_image, -np.inf, np.float64)
    best_score = per_unit(lambda p: p.relion_stats.best_log_score_per_image, -np.inf, np.float64)
    hard = per_unit(
        lambda p: sparse_pose_ids_to_fine_grid(
            p.hard_assignment, np.asarray(p.best_rotation_indices, dtype=np.int64), int(fine_px.shape[0])
        ),
        -1,
        np.int64,
    )
    max_posterior = per_unit(lambda p: p.relion_stats.max_posterior_per_image, 0.0, np.float64)
    rotation_sums = np.zeros((n_classes,) + np.shape(first.relion_stats.rotation_posterior_sums), dtype=np.float64)
    for k, r in present:
        rotation_sums[k] = np.asarray(r.pass2.relion_stats.rotation_posterior_sums, dtype=np.float64)
    units = np.arange(n_units)
    zero_y, zero_ctf = np.zeros_like(np.asarray(first.Ft_y)), np.zeros_like(np.asarray(first.Ft_ctf))
    by_class = dict(present)
    pass2 = ResidentKClassPass2Output(
        Ft_y=tuple(by_class[k].pass2.Ft_y if k in by_class else zero_y for k in range(n_classes)),
        Ft_ctf=tuple(by_class[k].pass2.Ft_ctf if k in by_class else zero_ctf for k in range(n_classes)),
        class_log_evidence_per_image=log_evidence,
        class_best_log_score_per_image=best_score,
        per_class_hard_assignments=hard,
        stats=make_relion_stats(
            log_evidence_per_image=log_evidence[unit_class, units],
            best_log_score_per_image=best_score[unit_class, units],
            max_posterior_per_image=max_posterior[unit_class, units],
            rotation_posterior_sums=rotation_sums.sum(axis=0),
        ),
        class_rotation_posterior_sums=rotation_sums,
        class_reconstruction_posterior_sums=np.asarray(
            [float(np.sum(np.asarray(by_class[k].pass2.noise_stats.sumw))) if k in by_class else 0.0 for k in range(n_classes)],
            dtype=np.float64,
        ),
        noise_stats=_sum_noise_stats(tuple(
            _expand_subset_noise_stats(r.pass2.noise_stats, members[k], n_units, full_group_count=scale_group_count)
            for k, r in present
        )),
        per_class_best_pose_rotations=per_class(lambda p: p.best_rotations, first.best_rotations),
        per_class_best_pose_translations=per_class(lambda p: p.best_translations, first.best_translations),
        per_class_best_pose_rotation_ids=per_class(
            lambda p: np.asarray(p.best_rotation_indices, dtype=np.int64),
            np.asarray(first.best_rotation_indices, dtype=np.int64),
        ),
        per_class_best_pose_eulers_deg=per_class(lambda p: p.source_eulers, first.source_eulers),
    )
    coarse_hard = np.zeros(n_units, dtype=np.int32)
    significant_counts = np.zeros(n_units, dtype=np.int32)
    for k, r in present:
        coarse_hard[members[k]] = np.asarray(r.coarse_hard_assignment, dtype=np.int32)
        significant_counts[members[k]] = np.asarray(r.significant_counts, dtype=np.int32)
    return TomoScoreResult(pass2=pass2, coarse_hard_assignment=coarse_hard, significant_counts=significant_counts)


def _first_class_k_class_output(pass2, n_classes: int, sampling: TomoSampling, pixel_size: float):
    """A K=1 tilt pass as the K-class pass with every candidate in class 0 (``ResidentKClassPass2Output``).

    What the K-class pass of a single-class iteration returns: the other classes have no candidates,
    so their evidence and best scores are ``-inf``, their poses ``-1`` and their accumulators and sums zero.
    """

    from relax.classification.k_class import sparse_pose_ids_to_fine_grid
    from relax.fine_pass.resident_pass2 import ResidentKClassPass2Output

    _, _, fine_px, _ = tomo_translation_grids(sampling, pixel_size)
    stats = pass2.relion_stats
    others = n_classes - 1

    def by_class(first, fill):
        first = np.asarray(first)
        return np.concatenate([first[None], np.full((others,) + first.shape, fill, dtype=first.dtype)])

    def class_tuple(first):
        return None if first is None else (first,) + tuple(np.zeros_like(np.asarray(first)) for _ in range(others))

    rotation_ids = np.asarray(pass2.best_rotation_indices, dtype=np.int64)
    hard = sparse_pose_ids_to_fine_grid(pass2.hard_assignment, rotation_ids, int(fine_px.shape[0]))
    rotation_sums = np.asarray(stats.rotation_posterior_sums, dtype=np.float64)
    return ResidentKClassPass2Output(
        Ft_y=(pass2.Ft_y,) + tuple(np.zeros_like(np.asarray(pass2.Ft_y)) for _ in range(others)),
        Ft_ctf=(pass2.Ft_ctf,) + tuple(np.zeros_like(np.asarray(pass2.Ft_ctf)) for _ in range(others)),
        class_log_evidence_per_image=by_class(np.asarray(stats.log_evidence_per_image, dtype=np.float64), -np.inf),
        class_best_log_score_per_image=by_class(
            np.asarray(stats.best_log_score_per_image, dtype=np.float64), -np.inf
        ),
        per_class_hard_assignments=by_class(np.asarray(hard, dtype=np.int64), -1),
        stats=stats,
        class_rotation_posterior_sums=by_class(rotation_sums, 0.0),
        class_reconstruction_posterior_sums=np.asarray(
            [float(pass2.noise_stats.sumw)] + [0.0] * others, dtype=np.float64
        ),
        noise_stats=pass2.noise_stats,
        per_class_best_pose_rotations=class_tuple(pass2.best_rotations),
        per_class_best_pose_translations=class_tuple(pass2.best_translations),
        per_class_best_pose_rotation_ids=class_tuple(rotation_ids),
        per_class_best_pose_eulers_deg=class_tuple(pass2.source_eulers),
    )


def score_tomo_half_in_loop(
    data: HalfScoringData,
    *,
    use_adaptive: bool,
    sampling: TomoSampling,
    rotation_log_prior,
    sigma_offset_angst: float,
    max_significants,
    reconstruction_current_size,
    local_search=None,
    symmetry: str = "C1",
    class_log_priors=None,
    class_rotation_log_prior=None,
    unit_seed_classes=None,
    normalized_cc: bool = False,
):
    """The refinement loop's E+M step for a tomo half: :func:`score_tomo_half` as a ``HalfScoreResult``.

    ``data`` is the half's expectation operands, as the SPA scorers take them: its particles (a
    ``TomoHalf`` dataset with their previous translations, optics groups and scale corrections), reference,
    noise, RELION projector and scale groups.

    Per-unit fields are the particles'. The best translations are the winning trial shifts in pixels
    (3D); the loop adds the rounded previous offset, as RELION writes ``old + shift``. The returned
    best poses and class summaries are recorded by the controller.

    A local-search iteration passes ``local_search``: a dict with the particles' previous angles
    (``previous_eulers_deg``), ``sigma_rot`` and ``sigma_psi`` and the pass-1 HEALPix order
    (``parent_order``); ``sampling`` then carries that order and the local grid's perturbation.

    Class3D (K>1, a global search): ``volume`` and ``relion_projector_half`` keep the loop's class axis,
    ``class_log_priors`` are the classes' ``log pdf_class`` and ``class_rotation_log_prior`` their
    direction priors ``[K, R]`` (or None: the shared ``rotation_log_prior``), folded as the SPA K-class
    pass folds them (``k_class.rotation_prior_with_class_log_prior``). Class outputs use the same SPA result adaptation
    (``_class_segmented_em_result``, ``class_em_to_half_result``). ``unit_seed_classes`` are the particles' classes in RELION's
    first iteration from one reference (:func:`score_tomo_half`). ``normalized_cc`` is the
    ``--firstiter_cc`` iteration (:func:`score_tomo_half`).
    """

    from relax.reconstruction.half_volume_mstep import relion_backprojector_volume_shape
    from relax.refinement.score_outputs import HalfScoreResult
    from relax.sampling import rotation_grid_size

    half = data.particles.dataset
    volume, noise_variance = data.reference, data.noise_variance
    relion_projector_half = None if data.projector is None else data.projector.data
    relion_projector_r_max = None if data.projector is None else data.projector.r_max
    previous_translations = data.particles.translations
    unit_groups, scale_corrections = data.particles.optics_group_ids, data.particles.scale_corrections
    group_ids, scale_correction_group_count = data.scale_group_ids, data.scale_group_count
    scale_correction_data_vs_prior = data.scale_correction_data_vs_prior
    if not (use_adaptive or local_search is not None) or int(sampling.oversampling_order) < 1:
        raise NotImplementedError("subtomogram particles run RELION's adaptive two-pass E-step only")
    if RECONSTRUCTION_PADDING_FACTOR != PROJECTION_PADDING_FACTOR:
        raise ValueError("the tomo half pass projects and backprojects with one padding factor")
    if relion_projector_half is None or relion_projector_r_max is None:
        raise ValueError("the tomo half pass needs RELION's Projector::data half map")
    # RELION's float projector (score_tomo_half).
    relion_projector_half = np.asarray(relion_projector_half, dtype=np.complex64)
    n_classes = 1 if class_log_priors is None else int(np.asarray(class_log_priors).size)
    if relion_projector_half.ndim == 4:
        # The loop keeps a class axis.
        if relion_projector_half.shape[0] != n_classes:
            raise ValueError(f"{relion_projector_half.shape[0]} projector halves for {n_classes} classes")
        if n_classes == 1:
            relion_projector_half = relion_projector_half[0]
    elif n_classes > 1:
        raise ValueError("a K-class tomo pass needs one projector half per class")
    r_max_by_class = np.asarray(relion_projector_r_max).reshape(-1)
    if np.unique(r_max_by_class).size != 1:
        raise ValueError(f"the classes' projectors must share one r_max, got {r_max_by_class}")
    relion_projector_r_max = r_max_by_class[0]
    if np.ndim(volume) == 2:
        if np.shape(volume)[0] != n_classes:
            raise ValueError(f"{np.shape(volume)[0]} class volumes for {n_classes} classes")
        if n_classes == 1:
            volume = volume[0]
    local_rotations = None
    if local_search is not None:
        local_rotations = tomo_local_rotations(
            local_search["previous_eulers_deg"],
            sigma_rot=local_search["sigma_rot"],
            sigma_psi=local_search["sigma_psi"],
            healpix_order=int(sampling.healpix_order),
            random_perturbation=float(sampling.random_perturbation),
            voxel_size=half.voxel_size,
            symmetry=symmetry,
        )
        prior = None
    else:
        n_rot = int(rotation_grid_size(sampling.healpix_order, symmetry))
        prior = (
            np.zeros(n_rot, dtype=np.float32)
            if rotation_log_prior is None
            else np.asarray(rotation_log_prior, dtype=np.float32).reshape(-1)
        )
        if prior.shape != (n_rot,):
            raise ValueError(f"a global tomo pass needs one rotation log prior per coarse rotation, got {prior.shape}")
        if n_classes > 1:
            from relax.classification.k_class import rotation_prior_with_class_log_prior

            class_priors = (
                [prior] * n_classes
                if class_rotation_log_prior is None
                else list(np.asarray(class_rotation_log_prior, dtype=np.float32).reshape(n_classes, n_rot))
            )
            prior = np.stack(
                [
                    rotation_prior_with_class_log_prior(class_prior, float(log_pdf), n_rot)
                    for class_prior, log_pdf in zip(class_priors, np.asarray(class_log_priors).reshape(-1))
                ]
            )
    old = (
        np.zeros((half.n_units, 3), dtype=np.float64)
        if previous_translations is None
        else np.asarray(previous_translations, dtype=np.float64)
    )
    if old.shape != (half.n_units, 3):
        raise ValueError(f"tomo particles carry 3D offsets, got previous translations of shape {old.shape}")
    groups = np.zeros(half.n_units, dtype=np.int32) if unit_groups is None else np.asarray(unit_groups, dtype=np.int32)
    result = score_tomo_half(
        half,
        volume=volume,
        noise_variance=noise_variance,
        relion_projector_half=relion_projector_half,
        relion_projector_r_max=int(relion_projector_r_max),
        sampling=sampling,
        rotation_log_prior=prior,
        old_offsets_px=old,
        sigma_offset_angst=sigma_offset_angst,
        adaptive_fraction=RELION_ADAPTIVE_FRACTION,
        max_significants=max_significants,
        unit_groups=groups,
        padding_factor=PROJECTION_PADDING_FACTOR,
        scale_corrections=scale_corrections,
        group_ids=group_ids,
        scale_correction_group_count=scale_correction_group_count,
        scale_correction_data_vs_prior=scale_correction_data_vs_prior,
        reconstruction_current_size=reconstruction_current_size,
        local_rotations=local_rotations,
        symmetry=symmetry,
        unit_seed_classes=unit_seed_classes,
        normalized_cc=normalized_cc,
    )
    pass2 = result.pass2
    mstep_size = sampling.fine_size if reconstruction_current_size is None else int(reconstruction_current_size)
    mstep_accumulator_shape = relion_backprojector_volume_shape(
        half.volume_shape, RECONSTRUCTION_PADDING_FACTOR, current_size=mstep_size
    )
    if n_classes > 1:
        from relax.classification.k_class import _class_segmented_em_result
        from relax.refinement.score_outputs import class_em_to_half_result

        k_class_result = _class_segmented_em_result(
            pass2,
            n_classes=n_classes,
            class_posterior_sums_from_noise=True,
            return_profile=False,
            host_accumulators=True,
            mstep_full_half_axis=0,
            mstep_accumulator_shape=mstep_accumulator_shape,
        )
        score_result = class_em_to_half_result(
            k_class_result,
            # The class rotation sums are already over the coarse grid.
            effective_rotations=np.zeros((int(rotation_grid_size(sampling.healpix_order, symmetry)), 0)),
            rot_pmap_for_collapse=None,
            adaptive_os_local=int(sampling.oversampling_order),
            require_best_pose_details=True,
            pose_dtype=np.float32,
        )
        score_result.ha = np.asarray(score_result.ha, dtype=np.int32)
        score_result.coarse_ha = result.coarse_hard_assignment
        score_result.significant_counts = result.significant_counts
        score_result.mstep_full_half_axis = 0
        score_result.mstep_accumulator_shape = mstep_accumulator_shape
        return score_result
    best_rotations = np.asarray(pass2.best_rotations, dtype=np.float32)
    best_eulers = np.asarray(pass2.source_eulers, dtype=np.float64)
    best_translations = np.asarray(pass2.best_translations, dtype=np.float32)
    return HalfScoreResult(
        ha=np.asarray(pass2.hard_assignment, dtype=np.int32),
        Ft_y=pass2.Ft_y,
        Ft_ctf=pass2.Ft_ctf,
        em_stats=pass2.relion_stats,
        noise_stats=pass2.noise_stats,
        best_pose_rotations=best_rotations,
        best_pose_rotation_eulers=best_eulers,
        best_pose_translations=best_translations,
        coarse_ha=result.coarse_hard_assignment,
        significant_counts=result.significant_counts,
        mstep_full_half_axis=0,
        mstep_accumulator_shape=mstep_accumulator_shape,
    )
