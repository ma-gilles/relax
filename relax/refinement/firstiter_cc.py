"""RELION first-iteration adaptive winner-take-all dispatch.

Batch budgets and adaptive pass plans are owned by ``batch_planning``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from relax.classification.k_class import run_dense_k_class_em_adaptive
from relax.classification.k_class_results import KClassEMResult
from relax.helpers.batch_planning import (
    _plan_kclass_adaptive_grid_batch_sizes,
    safe_dense_k_class_rotation_block_size,
    safe_firstiter_cc_image_batch_size,
)
from relax.helpers.oversampling import build_adaptive_pass2_grids, project_pass2_rotations
from relax.sampling import (
    apply_relion_translation_perturbation,
)
from relax.symmetry import canonicalize_rotational_symmetry

if TYPE_CHECKING:
    from relax.refinement.half_scoring import (
        DenseBatchPolicy,
        DenseExecutionPolicy,
        DensePriorSpec,
        DenseSamplingSpec,
        DenseVariantPolicy,
        HalfScoringData,
    )


# The dispatch is a step of dense half scoring and logs under its logger.
logger = logging.getLogger("relax.dense.half_scoring")


@dataclass(frozen=True, kw_only=True)
class FirstIterCCPass2:
    """The engine's result of a first-iteration CC dispatch and the trial-grid maps it ran on."""

    result: KClassEMResult
    # Each fine rotation's and fine translation's coarse parent.
    rotation_parent_map: np.ndarray
    translation_parent_map: np.ndarray
    n_fine_translations: int


def _score_kclass_firstiter_cc_pass2(
    half: HalfScoringData,
    sampling: DenseSamplingSpec,
    priors: DensePriorSpec,
    batching: DenseBatchPolicy,
    variant: DenseVariantPolicy,
    execution: DenseExecutionPolicy,
    *,
    projection_scale: float,
    magnification,
    em_kwargs: dict,
) -> FirstIterCCPass2:
    """RELION iter-1 ``--firstiter_cc`` adaptive two-pass dispatch.

    Build coarse/fine grids and invoke the K-class engine with normalized-CC
    scoring through the global coarse winner subset. The winner-take-all
    policy applies to M-step support as well as reported Pmax.

    The K-class and K=1 adaptive scoring branches share this dispatcher, with the images' projection scale
    and magnification and their engine keywords. A K-class half scores its class stack on its coarse rotation
    ids and logs under ``variant.firstiter_log_label``; a K=1 half scores its one reference as a one-class
    stack on every coarse rotation. The engine receives a clamped copy of ``em_kwargs``; the caller's
    dictionary is not changed. Coarse/fine size overrides (``variant.firstiter_*_current_size``) are
    forwarded only when supplied. The oversampling order used is ``sampling.oversampling_order``.
    """

    mean = half.reference if variant.k_class_enabled else jnp.asarray(half.reference)[None, :]
    # The coarse rotations a K-class half scores (None: all of them).
    coarse_ids = sampling.coarse_rotation_ids if variant.k_class_enabled else None
    log_label = variant.firstiter_log_label if variant.k_class_enabled else "K=1 "
    point_group = canonicalize_rotational_symmetry(sampling.symmetry)
    adaptive_os_local = int(sampling.oversampling_order)
    (
        coarse_rot,
        coarse_trans,
        fine_rot,
        fine_trans,
        rot_pmap,
        trans_pmap,
        fine_mstep_rot,
    ) = build_adaptive_pass2_grids(
        sampling.effective_rotations,
        sampling.current_translations,
        sampling.base_translations,
        int(sampling.current_healpix_order),
        adaptive_os_local,
        float(sampling.translation_step),
        sampling.random_perturbation,
        return_mstep_rotations=True,
        **(
            {"coarse_rotation_ids": coarse_ids}
            if coarse_ids is not None
            else {}
        ),
        symmetry=point_group,
    )
    coarse_rot, fine_rot, fine_mstep_rot = project_pass2_rotations(
        coarse_rot,
        fine_rot,
        fine_mstep_rot,
        scale=float(projection_scale),
        magnification=magnification,
        coarse_healpix_order=int(sampling.current_healpix_order),
        adaptive_oversampling=adaptive_os_local,
        random_perturbation=sampling.random_perturbation,
        coarse_rotation_ids=coarse_ids,
        symmetry=point_group,
        coarse_device_source=sampling.effective_device_source,
        grid_device_source=sampling.effective_device_source,
    )
    coarse_translation_phase_source = apply_relion_translation_perturbation(
        np.asarray(sampling.base_translations, dtype=np.float64),
        float(sampling.random_perturbation),
        float(sampling.translation_step),
    )
    n_classes = int(np.shape(mean)[0]) if np.ndim(mean) >= 2 else 1
    firstiter_significance_image_batch_size = None
    firstiter_significance_rotation_block_size = None
    if point_group != "C1" and em_kwargs.get("coarse_engine") != "gemm_dense":
        if not em_kwargs.get("mstep_relion_x_half", False):
            raise RuntimeError(f"{point_group} requires sparse RELION x-half BPref reconstruction")
    if batching.safe_batch_sizes is not None:
        batch_plan = _plan_kclass_adaptive_grid_batch_sizes(
            coarse_rotations=coarse_rot,
            coarse_translations=coarse_trans,
            fine_rotations=fine_rot,
            fine_translations=fine_trans,
            n_classes=n_classes,
            image_shape=half.particles.dataset.image_shape,
            coarse_current_size=(
                variant.firstiter_coarse_current_size
                if variant.firstiter_coarse_current_size is not None
                else em_kwargs.get("current_size")
            ),
            fine_current_size=(
                variant.firstiter_fine_current_size
                if variant.firstiter_fine_current_size is not None
                else em_kwargs.get("current_size")
            ),
            safe_batch_sizes=batching.safe_batch_sizes,
            significance_safe_batch_sizes=batching.significance_safe_batch_sizes,
        )
        requested_firstiter_image_batch_size = int(
            em_kwargs.get("image_batch_size", batching.image_batch_size)
        )
        firstiter_image_batch_size = min(
            requested_firstiter_image_batch_size,
            safe_firstiter_cc_image_batch_size(
                fine_trans.shape[0],
                half.particles.dataset.image_shape,
            ),
        )
        firstiter_rotation_block_size = min(
            int(em_kwargs.get("rotation_block_size", batch_plan.pass2_rotation_block_size)),
            safe_dense_k_class_rotation_block_size(
                fine_trans.shape[0],
                firstiter_image_batch_size,
            ),
        )
        firstiter_significance_image_batch_size = batch_plan.significance_image_batch_size
        firstiter_significance_rotation_block_size = batch_plan.significance_rotation_block_size
        logger.info(
            "STRICT-PARITY: iter-1 K-class adaptive batch sizing "
            "coarse image_batch_size=%d rotation_block_size=%d; "
            "fine image_batch_size=%d rotation_block_size=%d (sparse pass2)",
            firstiter_significance_image_batch_size,
            firstiter_significance_rotation_block_size,
            firstiter_image_batch_size,
            firstiter_rotation_block_size,
        )
    else:
        requested_firstiter_image_batch_size = int(
            em_kwargs.get("image_batch_size", batching.image_batch_size)
        )
        firstiter_image_batch_size = min(
            requested_firstiter_image_batch_size,
            safe_firstiter_cc_image_batch_size(
                fine_trans.shape[0],
                half.particles.dataset.image_shape,
            ),
        )
        firstiter_rotation_block_size = int(em_kwargs.get("rotation_block_size", 5000))
        if firstiter_image_batch_size != requested_firstiter_image_batch_size:
            logger.info(
                "STRICT-PARITY: clamping iter-1 winner-take-all image_batch_size from %d to %d",
                requested_firstiter_image_batch_size,
                firstiter_image_batch_size,
            )
    firstiter_em_kwargs = dict(em_kwargs)
    firstiter_em_kwargs["image_batch_size"] = firstiter_image_batch_size
    firstiter_em_kwargs["rotation_block_size"] = firstiter_rotation_block_size
    firstiter_em_kwargs["sparse_pass2"] = True
    logger.info(
        "STRICT-PARITY %srouting iter-1 K-class through sparse run_dense_k_class_em_adaptive "
        "(oversampling=%d, relion_x_half_mstep=%s, best_coarse_subset=True)",
        log_label,
        adaptive_os_local,
        bool(firstiter_em_kwargs.get("mstep_relion_x_half", False)),
    )
    extra: dict = {}
    if variant.firstiter_coarse_current_size is not None:
        extra["coarse_current_size"] = variant.firstiter_coarse_current_size
    if variant.firstiter_fine_current_size is not None:
        extra["fine_current_size"] = variant.firstiter_fine_current_size
    k_class_result = run_dense_k_class_em_adaptive(
        half.particles.dataset,
        mean,
        half.mean_variance,
        half.noise_variance,
        coarse_rot,
        coarse_trans,
        fine_rot,
        fine_trans,
        rot_pmap,
        trans_pmap,
        execution.disc_type,
        class_log_priors=priors.class_log_priors,
        accumulate_noise=True,
        return_best_pose_details=True,
        firstiter_cc_pass2_only_best_coarse=True,
        image_seed_classes=half.image_seed_classes,
        relion_fine_mstep_prune=True,
        significance_image_batch_size=firstiter_significance_image_batch_size,
        significance_rotation_block_size=firstiter_significance_rotation_block_size,
        coarse_healpix_order=int(sampling.current_healpix_order),
        coarse_rotation_ids=coarse_ids,
        oversampling_order=int(adaptive_os_local),
        fine_mstep_rotations_override=(
            fine_mstep_rot
        ),
        bpref_device_signature_active=execution.bpref_device_signature_active,
        debug_iteration=execution.debug_iteration,
        coarse_translation_phase_source=(
            coarse_translation_phase_source if n_classes == 1 else None
        ),
        **extra,
        **firstiter_em_kwargs,
    )
    return FirstIterCCPass2(
        result=k_class_result,
        rotation_parent_map=rot_pmap,
        translation_parent_map=trans_pmap,
        n_fine_translations=int(fine_trans.shape[0]),
    )
