"""K-class supports and classes at given poses (RELION ``--skip_align``), in place of pass 1.

``given_pose_class_supports`` builds pass 2's per-class supports from each image's given samples (with the
compacted CSR pass 1 would have produced); ``given_pose_cc_classes`` and ``classes_of_best_given_pose_cc``
find each image's class by the best normalized CC of its given pose in a ``--firstiter_cc`` iteration from
one reference per class. The engine entry (``k_class.run_dense_k_class_em_adaptive``) selects them.
"""

from __future__ import annotations

import logging

import numpy as np

from relax.classification.k_class_inputs import seed_iteration_supports

logger = logging.getLogger("relax.classification.k_class")  # the engine's logger before the split: log rows keep it


def _given_support_csr(class_supports, *, n_coarse_rot: int, n_coarse_trans: int):
    """A class's given per-image sample lists with the compacted CSR pass 1 would have produced.

    The resident pass reads the CSR for its candidate tables and to cache only the rotations the
    supports use (``resident_projections.significant_projection_slots``); a plain list makes it project the
    whole grid.
    """

    from relax.fine_pass.resident_significance import (
        DeviceCompactedSignificantSamples,
        build_coarse_significance_csr,
    )

    rows = [np.unique(np.asarray(samples, dtype=np.int32).reshape(-1)) for samples in class_supports]
    csr = build_coarse_significance_csr(
        n_images=len(rows),
        n_coarse_rot=int(n_coarse_rot),
        n_coarse_trans=int(n_coarse_trans),
        n_significant_per_batch=[np.asarray([row.size for row in rows], dtype=np.int32)],
        store_excluded_per_batch=[np.zeros(len(rows), dtype=bool)],
        ids_per_batch=[np.concatenate(rows) if rows else np.zeros(0, np.int32)],
    )
    return DeviceCompactedSignificantSamples(csr=csr)


def given_pose_class_supports(
    given_supports, *, n_classes, n_images, image_seed_classes, firstiter_cc, n_coarse_rot, n_coarse_trans
):
    """Pass 2's per-class supports at given poses (RELION ``--skip_align``), in place of pass 1.

    No pose search, so no pass 1 (ml_optimiser.cpp:8454, pdf_orientation = pdf_class): every class is
    scored on each image's own sample, or only its seed class in a seed iteration. RELION's
    ``--firstiter_cc`` iteration from one reference scores every particle against it alone
    (:4389-4402), so its winner at a given pose is the seed class; from one reference per class it scores
    every class and the caller finds the winner (:func:`given_pose_cc_classes`). Returns the supports by
    class, each image's sample count and, for the seeded cross-correlation iteration, each image's class.
    """

    if n_classes == 1 or len(given_supports) != n_images:
        raise ValueError("given supports are one coarse sample list per image of a K-class pass")
    supports = [np.asarray(samples, dtype=np.int32).reshape(-1) for samples in given_supports]
    by_class = (
        [list(supports) for _ in range(n_classes)]
        if image_seed_classes is None
        else seed_iteration_supports(supports, image_seed_classes, n_classes)
    )
    return (
        [_given_support_csr(rows, n_coarse_rot=n_coarse_rot, n_coarse_trans=n_coarse_trans) for rows in by_class],
        np.asarray([samples.size for samples in supports], dtype=np.int32),
        np.asarray(image_seed_classes, dtype=np.int32) if firstiter_cc and image_seed_classes is not None else None,
    )


def classes_of_best_given_pose_cc(cc_by_class) -> np.ndarray:
    """Each image's class in a ``--firstiter_cc`` iteration at given poses from K references (relax#78).

    RELION takes the device arg-min of the float32 ``diff2 = -CC`` over one class-major weight array
    (acc_ml_optimiser_impl.h:1384-1392 and 2012-2026; cuda_utils_cub.cuh:66-84), so the class of the best
    normalized CC wins and equal scores go to the lowest class. ``cc_by_class`` is ``[K, N]``, each image's
    float32 CC of every class, or that plus an offset of the image's own (the pass's evidence): the offset is
    0.5 |image|^2, about 2^21, and a float64 keeps a float32 CC below 1 added to it apart from its neighbours,
    so the order and the exact ties are the float32 CC's.
    """

    return np.argmax(np.asarray(cc_by_class, dtype=np.float64), axis=0).astype(np.int32)


def given_pose_cc_classes(score_class, n_classes: int, n_images: int) -> np.ndarray:
    """The winner class of each image's given pose under normalized CC, from one reference per class.

    Nothing restricts RELION's ``--firstiter_cc`` iteration to one reference under ``--skip_align``
    (``do_generate_seeds`` alone does, ml_optimiser.cpp:4392-4401), so each particle's one sample is scored
    against every class. ``score_class(assignments, evidence)`` is the winner-subset pass, here with every
    image assigned one class; it fills that class's row of ``evidence``, at one sample per image the sample's
    normalized CC less an offset of the image's own.
    """

    cc = np.full((n_classes, n_images), -np.inf, dtype=np.float64)
    for class_index in range(n_classes):
        score_class(np.full(n_images, class_index, dtype=np.int32), cc)
    if not np.all(np.isfinite(cc)):
        raise RuntimeError("a given-pose normalized CC is not finite")
    winners = classes_of_best_given_pose_cc(cc)
    logger.info(
        "RELION --skip_align --firstiter_cc from %d references: images per class %s",
        n_classes, np.bincount(winners, minlength=n_classes).tolist(),
    )
    return winners
