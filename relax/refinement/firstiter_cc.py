"""RELION first-iteration adaptive winner-take-all dispatch.

Batch budgets and adaptive pass plans are owned by ``batch_planning``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from relax.classification.k_class import run_dense_k_class_em_adaptive
from relax.helpers.batch_planning import (
    _plan_kclass_adaptive_grid_batch_sizes,
    _safe_dense_k_class_rotation_block_size,
    _safe_firstiter_cc_image_batch_size,
)
from relax.helpers.oversampling import build_adaptive_pass2_grids, project_pass2_rotations
from relax.sampling import (
    apply_relion_translation_perturbation,
)


@dataclass(frozen=True, kw_only=True)
class FirstIterCCData:
    """Half-set arrays and logging context for first-iteration CC scoring."""

    logger: logging.Logger
    experiment_dataset: object
    mean: object
    mean_variance: object
    noise_variance: object
    image_shape: tuple[int, ...]
    # A one-reference Class3D run's CC iteration scores every image against class 1 alone
    # (k_class_inputs.seed_iteration_supports).
    image_seed_classes: object | None = None


@dataclass(frozen=True, kw_only=True)
class FirstIterCCGridSpec:
    """Resolved coarse sampling and perturbation inputs."""

    effective_rotations: object
    current_translations: object
    base_translations: object
    current_healpix_order: int
    oversampling_order: int
    translation_step: float
    random_perturbation: float
    coarse_rotation_ids: object | None = None
    symmetry: str = "C1"
    # Images on another grid (applyScaleDifference) or magnified (applyAnisoMag): the grids' rows
    # become projection matrices by RELION's per-path rules (oversampling.project_pass2_rotations);
    # ``effective_device_source`` is the device pass-1 source of ``effective_rotations`` (None: host rows).
    projection_scale: float = 1.0
    magnification: object | None = None
    effective_device_source: object | None = None


@dataclass(frozen=True, kw_only=True)
class FirstIterCCPolicy:
    """Scoring and class-prior policy for the adaptive dispatch."""

    disc_type: str
    class_log_priors: object | None


@dataclass(frozen=True, kw_only=True)
class FirstIterCCBatching:
    """Caller batch limits and pass-size planners."""

    image_batch_size: int
    em_kwargs: dict
    safe_batch_sizes: object | None = None
    significance_safe_batch_sizes: object | None = None
    coarse_current_size: int | None = None
    fine_current_size: int | None = None


@dataclass(frozen=True, kw_only=True)
class FirstIterCCExecution:
    """Routing, mutation and diagnostic controls."""

    log_label: str = ""
    bpref_device_signature_active: bool = False
    debug_iteration: int | None = None


def _score_kclass_firstiter_cc_pass2(
    data: FirstIterCCData,
    grid: FirstIterCCGridSpec,
    policy: FirstIterCCPolicy,
    batching: FirstIterCCBatching,
    execution: FirstIterCCExecution,
):
    """RELION iter-1 ``--firstiter_cc`` adaptive two-pass dispatch.

    Build coarse/fine grids and invoke the K-class engine with normalized-CC
    scoring through the global coarse winner subset. The winner-take-all
    policy applies to M-step support as well as reported Pmax.

    The K-class and K=1 adaptive scoring branches share this dispatcher.
    The engine receives a clamped copy of ``batching.em_kwargs``; the caller's
    dictionary is not changed. Coarse/fine size overrides are forwarded only when supplied.
    ``log_label`` identifies the caller in routing messages.

    Return ``(k_class_result, rot_pmap, trans_pmap, n_trans_fine)``; the oversampling
    order used is the caller's ``grid.oversampling_order``.
    """

    adaptive_os_local = int(grid.oversampling_order)
    (
        coarse_rot,
        coarse_trans,
        fine_rot,
        fine_trans,
        rot_pmap,
        trans_pmap,
        fine_mstep_rot,
    ) = build_adaptive_pass2_grids(
        grid.effective_rotations,
        grid.current_translations,
        grid.base_translations,
        int(grid.current_healpix_order),
        adaptive_os_local,
        float(grid.translation_step),
        grid.random_perturbation,
        return_mstep_rotations=True,
        **(
            {"coarse_rotation_ids": grid.coarse_rotation_ids}
            if grid.coarse_rotation_ids is not None
            else {}
        ),
        symmetry=grid.symmetry,
    )
    coarse_rot, fine_rot, fine_mstep_rot = project_pass2_rotations(
        coarse_rot,
        fine_rot,
        fine_mstep_rot,
        scale=float(grid.projection_scale),
        magnification=grid.magnification,
        coarse_healpix_order=int(grid.current_healpix_order),
        adaptive_oversampling=adaptive_os_local,
        random_perturbation=grid.random_perturbation,
        coarse_rotation_ids=grid.coarse_rotation_ids,
        symmetry=grid.symmetry,
        coarse_device_source=grid.effective_device_source,
        grid_device_source=grid.effective_device_source,
    )
    coarse_translation_phase_source = apply_relion_translation_perturbation(
        np.asarray(grid.base_translations, dtype=np.float64),
        float(grid.random_perturbation),
        float(grid.translation_step),
    )
    n_classes = int(np.shape(data.mean)[0]) if np.ndim(data.mean) >= 2 else 1
    firstiter_significance_image_batch_size = None
    firstiter_significance_rotation_block_size = None
    if grid.symmetry != "C1" and batching.em_kwargs.get("coarse_engine") != "gemm_dense":
        if not batching.em_kwargs.get("mstep_relion_x_half", False):
            raise RuntimeError(f"{grid.symmetry} requires sparse RELION x-half BPref reconstruction")
    if batching.safe_batch_sizes is not None:
        batch_plan = _plan_kclass_adaptive_grid_batch_sizes(
            coarse_rotations=coarse_rot,
            coarse_translations=coarse_trans,
            fine_rotations=fine_rot,
            fine_translations=fine_trans,
            n_classes=n_classes,
            image_shape=data.image_shape,
            coarse_current_size=(
                batching.coarse_current_size
                if batching.coarse_current_size is not None
                else batching.em_kwargs.get("current_size")
            ),
            fine_current_size=(
                batching.fine_current_size
                if batching.fine_current_size is not None
                else batching.em_kwargs.get("current_size")
            ),
            safe_batch_sizes=batching.safe_batch_sizes,
            significance_safe_batch_sizes=batching.significance_safe_batch_sizes,
        )
        requested_firstiter_image_batch_size = int(
            batching.em_kwargs.get("image_batch_size", batching.image_batch_size)
        )
        firstiter_image_batch_size = min(
            requested_firstiter_image_batch_size,
            _safe_firstiter_cc_image_batch_size(
                fine_trans.shape[0],
                data.image_shape,
            ),
        )
        firstiter_rotation_block_size = min(
            int(batching.em_kwargs.get("rotation_block_size", batch_plan.pass2_rotation_block_size)),
            _safe_dense_k_class_rotation_block_size(
                fine_trans.shape[0],
                firstiter_image_batch_size,
            ),
        )
        firstiter_significance_image_batch_size = batch_plan.significance_image_batch_size
        firstiter_significance_rotation_block_size = batch_plan.significance_rotation_block_size
        data.logger.info(
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
            batching.em_kwargs.get("image_batch_size", batching.image_batch_size)
        )
        firstiter_image_batch_size = min(
            requested_firstiter_image_batch_size,
            _safe_firstiter_cc_image_batch_size(
                fine_trans.shape[0],
                data.image_shape,
            ),
        )
        firstiter_rotation_block_size = int(batching.em_kwargs.get("rotation_block_size", 5000))
        if firstiter_image_batch_size != requested_firstiter_image_batch_size:
            data.logger.info(
                "STRICT-PARITY: clamping iter-1 winner-take-all image_batch_size from %d to %d",
                requested_firstiter_image_batch_size,
                firstiter_image_batch_size,
            )
    firstiter_em_kwargs = dict(batching.em_kwargs)
    firstiter_em_kwargs["image_batch_size"] = firstiter_image_batch_size
    firstiter_em_kwargs["rotation_block_size"] = firstiter_rotation_block_size
    firstiter_em_kwargs["sparse_pass2"] = True
    data.logger.info(
        "STRICT-PARITY %srouting iter-1 K-class through sparse run_dense_k_class_em_adaptive "
        "(oversampling=%d, relion_x_half_mstep=%s, best_coarse_subset=True)",
        execution.log_label,
        adaptive_os_local,
        bool(firstiter_em_kwargs.get("mstep_relion_x_half", False)),
    )
    extra: dict = {}
    if batching.coarse_current_size is not None:
        extra["coarse_current_size"] = batching.coarse_current_size
    if batching.fine_current_size is not None:
        extra["fine_current_size"] = batching.fine_current_size
    k_class_result = run_dense_k_class_em_adaptive(
        data.experiment_dataset,
        data.mean,
        data.mean_variance,
        data.noise_variance,
        coarse_rot,
        coarse_trans,
        fine_rot,
        fine_trans,
        rot_pmap,
        trans_pmap,
        policy.disc_type,
        class_log_priors=policy.class_log_priors,
        accumulate_noise=True,
        return_best_pose_details=True,
        firstiter_cc_pass2_only_best_coarse=True,
        image_seed_classes=data.image_seed_classes,
        relion_fine_mstep_prune=True,
        significance_image_batch_size=firstiter_significance_image_batch_size,
        significance_rotation_block_size=firstiter_significance_rotation_block_size,
        coarse_healpix_order=int(grid.current_healpix_order),
        coarse_rotation_ids=grid.coarse_rotation_ids,
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
    return k_class_result, rot_pmap, trans_pmap, int(fine_trans.shape[0])
