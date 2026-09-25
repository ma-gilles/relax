"""InitialModel (VDAM) E-step through the shared adaptive pass-1/pass-2 route.

RELION's gradient E-step is the auto-refine E-step (``expectationOneParticle``:
the same coarse pass, significance and weighted sums) except in three places:

1. BPref backprojects the residual ``shift(img) - ctf * proj`` instead of the
   image (``cuda_kernel_backproject3D_SGD``, acc/cuda/cuda_kernels/BP.cuh:406-560,
   selected by ``do_grad`` at acc_ml_optimiser_impl.h:4827);
2. each class has one model and two pseudo-halfset BPref slots,
   ``iclass + (part_id % 2) * nr_classes`` (acc_ml_optimiser_impl.h:4800-4804);
3. ``maximum_significants`` defaults to ``100 * nr_classes`` (ml_optimiser.cpp:3692-3700)
   and caps the coarse pass only (acc_ml_optimiser_impl.h:3256-3260).

This module runs VDAM's E-step on ``run_dense_k_class_em_adaptive``, the route
auto-refine and Class3D take, on the device-resident pass 2. Difference 1 is the
route's ``mstep_subtract_ctf_projection``; 2 is the resident engine's accumulator
slots ``class + K * half`` (``reconstruction_group_ids``, one pass over the subset;
docs/development/resident_segments.md), which the compact engine refuses; 3 is the
``max_significants`` VDAM already resolves.

The route indexes coarse rotations in RECOVAR order (psi-slow,
direction-fast); VDAM's own state (orientation priors, ``pdf_direction``) uses
RELION's direction-major order. The permutation is applied at this boundary
only. Selected by ``--pass2_engine adaptive``; the exact-local VDAM route in
:mod:`relax.vdam.sparse_pass2_estep` stays the default until this one is
qualified, and is removed with it (transitional switch, listed in em_status).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from relax import sampling
from relax.classification.k_class import run_dense_k_class_em_adaptive
from relax.helpers.preprocessing import uses_relion_cuda_image_preprocessing
from relax.refinement.half_scoring import _adaptive_pass2_grids
from relax.scoring.sparse_bucket_arrays import relion_parent_execution_key
from relax.vdam.estep_common import (
    DenseInitialModelEstepConfig,
    DenseInitialModelEstepResult,
    _add_accumulator_weight_meta,
    _arrays_to_accumulators,
    _empty_accumulator,
    _group_local_kwargs,
    _select_image_rows,
)
from relax.vdam.sparse_pass2_estep import (
    _pop_sparse_pass2_options,
    _resolve_sparse_pass1_current_size,
    _safe_coarse_significance_image_batch_size,
    _sparse_pass2_estep_meta,
    _translation_step_from_grid,
)
from relax.vdam.state import InitialModelState

__all__ = ["AdaptiveRouteGrids", "adaptive_route_grids", "run_adaptive_initial_model_estep"]


def relion_order_of_recovar_rotations(healpix_order: int) -> np.ndarray:
    """RELION direction-major coarse id of each RECOVAR-order coarse rotation."""

    n_rot = int(sampling.rotation_grid_size(int(healpix_order)))
    return relion_parent_execution_key(
        np.arange(n_rot, dtype=np.int64), n_coarse_rot=n_rot, nside_level=int(healpix_order)
    )


class AdaptiveRouteGrids(NamedTuple):
    """One iteration's coarse and fine trial grids in the adaptive route's order."""

    pass1_rotations: np.ndarray
    grids: object  # relax.refinement.half_scoring._AdaptivePass2Grids
    fine_source_eulers: np.ndarray | None
    relion_of_recovar: np.ndarray


def adaptive_route_grids(
    *,
    healpix_order: int,
    oversampling_order: int,
    random_perturbation: float,
    coarse_base_translations: np.ndarray,
    translation_step: float,
) -> AdaptiveRouteGrids:
    """Build RELION's two-pass grids as auto-refine builds them (``_adaptive_pass2_grids``).

    Pass 1 scores RELION's device-built coarse matrices
    (``AccProjectorPlan::setup``, :func:`relax.sampling._relion_adaptive_pass1_rotations`);
    the fine and M-step rotations are host-generated from the unperturbed source
    Euler rows, and the translations are oversampled from the host-double base grid
    before the SamplingPerturbation shift.
    """

    order = int(healpix_order)
    angular_sampling = sampling.relion_angular_sampling_deg(order)
    source_eulers = sampling._get_relion_rotation_grid_eulers_float64(order, rotation_index_order="recovar")
    host_rotations = sampling.apply_relion_rotation_perturbation(
        sampling.get_relion_rotation_grid(order, rotation_index_order="recovar").astype(np.float32),
        float(random_perturbation),
        angular_sampling,
    ).astype(np.float32, copy=False)
    device_rotations = sampling._relion_adaptive_pass1_rotations(
        source_eulers, float(random_perturbation), angular_sampling
    )
    pass1_rotations = host_rotations if device_rotations is None else np.asarray(device_rotations)

    base = np.asarray(coarse_base_translations, dtype=np.float64)
    coarse_translations = sampling.apply_relion_translation_perturbation(
        base, float(random_perturbation), float(translation_step)
    ).astype(np.float32)
    grids = _adaptive_pass2_grids(
        pass1_rotations if int(oversampling_order) > 0 else host_rotations,
        coarse_translations,
        base,
        healpix_order=order,
        adaptive_oversampling=int(oversampling_order),
        translation_step=float(translation_step),
        random_perturbation=float(random_perturbation),
        coarse_rotation_ids=None,
    )
    n_rot = int(host_rotations.shape[0])
    fine_source_eulers = sampling.get_oversampled_rotation_grid_from_samples(
        np.arange(n_rot, dtype=np.int64),
        order,
        oversampling_order=int(oversampling_order),
        random_perturbation=float(random_perturbation),
        return_source_eulers=True,
    )[-1]
    if fine_source_eulers is not None and fine_source_eulers.shape[0] != grids.fine_rotations.shape[0]:
        raise RuntimeError("fine source Euler rows do not match the fine rotation grid")
    return AdaptiveRouteGrids(
        pass1_rotations=pass1_rotations,
        grids=grids,
        fine_source_eulers=fine_source_eulers,
        relion_of_recovar=relion_order_of_recovar_rotations(order),
    )


def _direction_posterior_stats(result, *, n_coarse_rot: int, rot_parent_map: np.ndarray, n_psi: int):
    """Bin each class's rotation posterior mass by HEALPix direction (``pdf_direction``).

    RECOVAR order is ``psi * n_directions + direction``, so a coarse id's
    direction is ``id % n_directions``. Fine sums are first collapsed onto
    their coarse parents.
    """

    n_directions = int(n_coarse_rot) // int(n_psi)
    binned = []
    for stats in result.per_class_stats:
        sums = np.asarray(stats.rotation_posterior_sums, dtype=np.float64).reshape(-1)
        if sums.size == rot_parent_map.size and sums.size != n_coarse_rot:
            sums = np.bincount(rot_parent_map, weights=sums, minlength=n_coarse_rot)
        if sums.size != n_coarse_rot:
            raise ValueError(
                f"rotation posterior sums have {sums.size} entries; expected the coarse grid "
                f"({n_coarse_rot}) or the fine grid ({rot_parent_map.size})"
            )
        direction_sums = np.bincount(np.arange(n_coarse_rot) % n_directions, weights=sums, minlength=n_directions)
        binned.append(stats._replace(rotation_posterior_sums=direction_sums))
    return result._replace(per_class_stats=tuple(binned))


def _recovar_order_prior(prior, relion_of_recovar: np.ndarray):
    if prior is None:
        return None
    prior = np.asarray(prior)
    if prior.shape[-1] != relion_of_recovar.size:
        raise ValueError(f"rotation prior has {prior.shape[-1]} coarse entries; expected {relion_of_recovar.size}")
    return prior[..., relion_of_recovar]


def run_adaptive_initial_model_estep(
    experiment_dataset,
    state: InitialModelState,
    config: DenseInitialModelEstepConfig,
    *,
    class_log_priors,
    joint_particle_ids: np.ndarray,
    joint_halfset_ids: np.ndarray | None,
    means,
    mean_variance,
    relion_projector_half_by_class,
    relion_projector_r_max,
    engine_kwargs: dict[str, Any],
) -> DenseInitialModelEstepResult:
    """Run one VDAM E-step as one adaptive-route pass over the subset, both pseudo-halfsets at once."""

    base_kwargs, options = _pop_sparse_pass2_options(engine_kwargs)
    if relion_projector_half_by_class is None:
        raise ValueError("the adaptive InitialModel route requires the exact RELION projector")
    if not config.relion_bpref_frame:
        raise ValueError("the adaptive InitialModel route writes RELION BPref accumulators")
    healpix_order = int(options.get("healpix_order", 1))
    oversampling_order = int(options.get("oversampling_order", 1))
    random_perturbation = float(options.get("random_perturbation", 0.0))
    coarse_translations = np.asarray(options["coarse_translations"], dtype=np.float32)
    translation_step = float(options.get("translation_step", _translation_step_from_grid(coarse_translations)))
    coarse_base_translations = base_kwargs.pop("coarse_base_translations")
    route = adaptive_route_grids(
        healpix_order=healpix_order,
        oversampling_order=oversampling_order,
        random_perturbation=random_perturbation,
        coarse_base_translations=coarse_base_translations,
        translation_step=translation_step,
    )
    grids = route.grids
    if not np.allclose(grids.coarse_translations, coarse_translations, rtol=0.0, atol=1e-5):
        raise RuntimeError("the adaptive route's coarse translations differ from VDAM's sampling plan")
    if np.asarray(config.translations).shape != grids.fine_translations.shape or not np.allclose(
        grids.fine_translations, config.translations, rtol=0.0, atol=1e-5
    ):
        raise RuntimeError("the adaptive route's fine translations differ from VDAM's sampling plan")
    n_coarse_rot = int(grids.coarse_rotations.shape[0])
    n_psi = int(sampling.rotation_grid_n_in_planes(healpix_order))
    class_rotation_log_prior = _recovar_order_prior(
        base_kwargs.get("class_rotation_log_prior"), route.relion_of_recovar
    )
    rotation_log_prior = _recovar_order_prior(base_kwargs.get("rotation_log_prior"), route.relion_of_recovar)
    significance_image_batch_size = _safe_coarse_significance_image_batch_size(
        config.image_batch_size,
        n_classes=state.K,
        n_rotations=n_coarse_rot,
        n_translations=int(coarse_translations.shape[0]),
    )

    image_indices = np.asarray(joint_particle_ids, dtype=np.int64)
    grouped = bool(state.pseudo_halfsets)
    if image_indices.size == 0:
        empty = [_empty_accumulator(state, k, h) for h in ((0, 1) if grouped else (0,)) for k in range(state.K)]
        return DenseInitialModelEstepResult(accumulators=empty, meta={"pass2_engine": "adaptive"}, halfset_results={})
    group_ids = None
    if grouped:
        # Difference 2: one reference per class and two pseudo-halfset BPref slots,
        # iclass + (part_id % 2) * nr_classes (acc_ml_optimiser_impl.h:4800-4804), in
        # one pass over the subset; the subset schedule supplies each particle's half.
        group_ids = np.asarray(joint_halfset_ids, dtype=np.int32)
        if group_ids.shape != image_indices.shape or np.any((group_ids != 0) & (group_ids != 1)):
            raise ValueError("pseudo-halfset ids must give each selected particle 0 or 1")
    n_images_total = int(experiment_dataset.n_images)
    group_kwargs = _group_local_kwargs(base_kwargs, image_indices, n_images=n_images_total)
    group_dataset = experiment_dataset.subset(image_indices)
    coarse_translation_log_prior = _select_image_rows(
        options.get("coarse_translation_log_prior"),
        image_indices,
        n_images=n_images_total,
        name="coarse_translation_log_prior",
    )
    current_size = group_kwargs.get("current_size")
    pass1_current_size = (
        current_size
        if oversampling_order == 0
        else _resolve_sparse_pass1_current_size(state, group_kwargs, options)
    )
    fresh_k1 = bool(state.K == 1 and uses_relion_cuda_image_preprocessing(group_dataset))
    route_kwargs = dict(
        image_batch_size=int(config.image_batch_size),
        rotation_block_size=int(config.rotation_block_size),
        current_size=current_size,
        sparse_pass2=True,
        mstep_relion_x_half=True,
        # Difference 1: the SGD backprojection of the residual.
        mstep_subtract_ctf_projection=bool(group_kwargs["reconstruction_subtract_projected_reference"]),
        score_with_masked_images=bool(group_kwargs["score_with_masked_images"]),
        half_spectrum_scoring=bool(group_kwargs["half_spectrum_scoring"]),
        projection_padding_factor=int(group_kwargs["projection_padding_factor"]),
        reconstruction_padding_factor=int(group_kwargs["reconstruction_padding_factor"]),
        projection_mask_current_image_disk=bool(group_kwargs["projection_mask_current_image_disk"]),
        image_pre_shifts=group_kwargs.get("image_pre_shifts"),
        translation_prior_centers=group_kwargs.get("translation_prior_centers"),
        # RELION reuses the coarse pdf_offset for every oversampled child.
        translation_log_prior=coarse_translation_log_prior,
        class_rotation_log_prior=class_rotation_log_prior,
        rotation_log_prior=rotation_log_prior,
        relion_firstiter_score_mode="gaussian",
        relion_exact_fine_gaussian=True,
        fine_source_eulers_override=route.fine_source_eulers,
        # The fresh K=1 guard, as VDAM's exact-local route keeps it: RELION's BPref
        # particle order, exact BPref operands and powerClass spectrum norm.
        preserve_bpref_particle_order=fresh_k1,
        source_faithful_spectrum_norm=fresh_k1,
        debug_iteration=group_kwargs.get("debug_iteration"),
        reconstruction_group_ids=group_ids,
        reconstruction_group_count=2 if grouped else None,
    )
    route_kwargs = {name: value for name, value in route_kwargs.items() if value is not None}
    result = run_dense_k_class_em_adaptive(
        group_dataset,
        means,
        mean_variance,
        config.noise_variance,
        route.pass1_rotations,
        grids.coarse_translations,
        grids.fine_rotations,
        grids.fine_translations,
        grids.rotation_parent_map,
        grids.translation_parent_map,
        config.disc_type,
        class_log_priors=class_log_priors,
        accumulate_noise=True,
        adaptive_fraction=float(options.get("adaptive_fraction", 0.999)),
        # Difference 3: VDAM's resolved cap, applied to the coarse pass only.
        max_significants=int(options.get("max_significants", -1)),
        significance_image_batch_size=significance_image_batch_size,
        significance_rotation_block_size=int(config.rotation_block_size),
        significance_pad_final_image_batch=True,
        coarse_current_size=pass1_current_size,
        fine_current_size=current_size,
        coarse_healpix_order=healpix_order,
        oversampling_order=oversampling_order,
        coarse_translation_log_prior=coarse_translation_log_prior,
        relion_fine_mstep_prune=True,
        relion_projector_half=relion_projector_half_by_class,
        relion_projector_r_max=relion_projector_r_max,
        fine_mstep_rotations_override=grids.fine_mstep_rotations,
        return_best_pose_details=True,
        coarse_translation_phase_source=grids.coarse_translation_phase_source,
        **route_kwargs,
    )
    result = _direction_posterior_stats(
        result,
        n_coarse_rot=n_coarse_rot,
        rot_parent_map=np.asarray(grids.rotation_parent_map, dtype=np.int64),
        n_psi=n_psi,
    )
    accumulators = _arrays_to_accumulators(
        result.Ft_y,
        result.Ft_ctf,
        state,
        halfset_idx=None if grouped else 0,
        reconstruction_group_count=2 if grouped else None,
        relion_bpref_frame=True,
        relion_projector_frame=config.relion_projector_frame,
        padding_factor=config.padding_factor,
    )
    halfset_results = {0: result}
    selected = {0: image_indices}

    meta = _sparse_pass2_estep_meta(halfset_results, selected)
    # The route's rotation ids index its RECOVAR-order fine grid; VDAM reads rotation
    # ids as RELION-order rows. The source Euler rows and matrices carry the pose.
    meta.pop("best_pose_rotation_ids", None)
    _add_accumulator_weight_meta(meta, accumulators, state.K)
    meta["pass2_engine"] = "adaptive"
    if grouped:
        meta["halfset_ids"] = (0, 1)
        meta["joint_halfset_particle_stream"] = True
    return DenseInitialModelEstepResult(accumulators=accumulators, meta=meta, halfset_results=halfset_results)
