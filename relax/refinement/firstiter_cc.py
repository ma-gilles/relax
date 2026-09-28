"""RELION first-iteration adaptive winner-take-all dispatch.

Batch budgets and adaptive pass plans are owned by ``batch_planning``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from relax.classification.k_class import _sparse_pass2_selected, run_dense_k_class_em_adaptive
from relax.helpers.batch_planning import (
    _plan_kclass_adaptive_grid_batch_sizes,
    _safe_dense_k_class_rotation_block_size,
    _safe_firstiter_cc_image_batch_size,
)
from relax.helpers.oversampling import build_adaptive_pass2_grids
from relax.helpers.preprocessing import uses_relion_cuda_image_preprocessing
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


@dataclass(frozen=True, kw_only=True)
class FirstIterCCGridSpec:
    """Coarse sampling state and perturbation inputs."""

    effective_rotations: object
    current_translations: object
    base_translations: object
    current_healpix_order: int
    state: object
    random_perturbation: float
    coarse_rotation_ids: object | None = None
    symmetry: str = "C1"


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
    update_em_kwargs_image_batch_size: bool = False
    bpref_device_signature_active: bool = False
    debug_iteration: int | None = None


@dataclass(frozen=True, kw_only=True)
class FirstIterCCSpec:
    """Complete first-iteration CC adaptive-dispatch specification."""

    data: FirstIterCCData
    grid: FirstIterCCGridSpec
    policy: FirstIterCCPolicy
    batching: FirstIterCCBatching
    execution: FirstIterCCExecution


def single_class_bucketed_pass2_selected(*, firstiter: bool) -> bool:
    """Whether K=1 plans its batches for the sparse projector/BPref lifetime.

    Only the ``--firstiter_cc`` pass does (its compact batch planning), and only while the
    ``RELAX_K_CLASS_DENSE_PASS2`` diagnostic keeps the sparse pass 2.
    """
    return bool(firstiter and _sparse_pass2_selected("RELAX_K_CLASS_DENSE_PASS2"))


def _score_kclass_firstiter_cc_pass2(
    spec: FirstIterCCSpec,
):
    """RELION iter-1 ``--firstiter_cc`` adaptive two-pass dispatch.

    Build coarse/fine grids and invoke the K-class engine with normalized-CC
    scoring through the global coarse winner subset. The winner-take-all
    policy applies to M-step support as well as reported Pmax.

    The K-class and K=1 adaptive scoring branches share this dispatcher.
    ``update_em_kwargs_image_batch_size`` controls whether the batch clamp
    also updates the caller's dictionary; the engine receives a clamped copy
    either way. Coarse/fine size overrides are forwarded only when supplied.
    ``log_label`` identifies the caller in routing messages.

    Return ``(k_class_result, rot_pmap, trans_pmap, n_trans_fine, adaptive_os)``.
    """

    data = spec.data
    grid = spec.grid
    policy = spec.policy
    batching = spec.batching
    execution = spec.execution

    adaptive_os_local = int(grid.state.adaptive_oversampling)
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
        float(grid.state.translation_step),
        grid.random_perturbation,
        return_mstep_rotations=True,
        **(
            {"coarse_rotation_ids": grid.coarse_rotation_ids}
            if grid.coarse_rotation_ids is not None
            else {}
        ),
        **({"symmetry": grid.symmetry} if grid.symmetry != "C1" else {}),
    )
    coarse_translation_phase_source = apply_relion_translation_perturbation(
        np.asarray(grid.base_translations, dtype=np.float64),
        float(grid.random_perturbation),
        float(grid.state.translation_step),
    )
    n_classes = int(np.asarray(data.mean).shape[0]) if np.asarray(data.mean).ndim >= 2 else 1
    firstiter_significance_image_batch_size = None
    firstiter_significance_rotation_block_size = None
    firstiter_sparse_pass2 = _sparse_pass2_selected("RELAX_K_CLASS_DENSE_PASS2")
    if grid.symmetry != "C1":
        if not firstiter_sparse_pass2 or not batching.em_kwargs.get("mstep_relion_x_half", False):
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
        if firstiter_sparse_pass2:
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
        else:
            firstiter_image_batch_size = batch_plan.pass2_image_batch_size
            firstiter_rotation_block_size = batch_plan.pass2_rotation_block_size
        firstiter_significance_image_batch_size = batch_plan.significance_image_batch_size
        firstiter_significance_rotation_block_size = batch_plan.significance_rotation_block_size
        data.logger.info(
            "STRICT-PARITY: iter-1 K-class adaptive batch sizing "
            "coarse image_batch_size=%d rotation_block_size=%d; "
            "fine image_batch_size=%d rotation_block_size=%d (%s pass2)",
            firstiter_significance_image_batch_size,
            firstiter_significance_rotation_block_size,
            firstiter_image_batch_size,
            firstiter_rotation_block_size,
            "sparse" if firstiter_sparse_pass2 else "dense",
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
    if execution.update_em_kwargs_image_batch_size:
        batching.em_kwargs["image_batch_size"] = firstiter_image_batch_size
    firstiter_em_kwargs = dict(batching.em_kwargs)
    firstiter_em_kwargs["image_batch_size"] = firstiter_image_batch_size
    firstiter_em_kwargs["rotation_block_size"] = firstiter_rotation_block_size
    firstiter_em_kwargs["sparse_pass2"] = firstiter_sparse_pass2
    # The normalized-CC coarse probe scores RELION's exact coarse operands.
    firstiter_em_kwargs["relion_exact_coarse"] = uses_relion_cuda_image_preprocessing(data.experiment_dataset)
    data.logger.info(
        "STRICT-PARITY %srouting iter-1 K-class through %s run_dense_k_class_em_adaptive "
        "(oversampling=%d, relion_x_half_mstep=%s, best_coarse_subset=True)",
        execution.log_label,
        "sparse" if firstiter_sparse_pass2 else "dense",
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
        skip_significance_pruning=False,
        relion_fine_mstep_prune=True,
        significance_image_batch_size=firstiter_significance_image_batch_size,
        significance_rotation_block_size=firstiter_significance_rotation_block_size,
        coarse_healpix_order=int(grid.current_healpix_order),
        coarse_rotation_ids=grid.coarse_rotation_ids,
        oversampling_order=int(adaptive_os_local),
        fine_mstep_rotations_override=(fine_mstep_rot if firstiter_sparse_pass2 else None),
        bpref_device_signature_active=execution.bpref_device_signature_active,
        debug_iteration=execution.debug_iteration,
        coarse_translation_phase_source=(
            coarse_translation_phase_source if n_classes == 1 else None
        ),
        **extra,
        **firstiter_em_kwargs,
    )
    return k_class_result, rot_pmap, trans_pmap, int(fine_trans.shape[0]), adaptive_os_local
