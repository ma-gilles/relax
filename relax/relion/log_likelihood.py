"""RELION's per-particle log-likelihood contribution (``rlnLogLikeliContribution``, ``METADATA_DLL``).

relion_refine sets it in storeWeightedSums for every particle of every iteration (acc_ml_optimiser_impl.h:3555-3578,
ml_optimiser.cpp:9029-9058) and writes it to the iteration's data STAR; Refine3D, Class3D and InitialModel write
it from here.
"""

from __future__ import annotations

import numpy as np


def relion_log_likelihood_contributions(
    log_evidence, *, sigma2_noise, groups, n_images, box_size: int, current_size: int, group_grids=None
):
    """RELION's per-particle dLL, ``log(sum_weight) - min_diff2 - logsigma2`` (ml_optimiser.cpp:9029-9058).

    ``log_evidence`` is the E-step's ``log(sum_weight) - min_diff2`` per particle; ``logsigma2`` sums
    ``log(2 pi sigma2_noise[group][ires])`` over the current-size ``Mresol_fine`` pixels with ``ires > 0``
    once per image of the particle (``n_images``: 1, or a subtomogram's tilt images).

    ``group_grids`` (optics groups on several image shapes) maps a group to its ``(box, scale, remap)``:
    its ``Mresol_fine`` is its own image's at its remapped current size (``scale``, the group's
    ``remap_sizes``), and each ``ires`` reads the model shell ``ROUND(remap * ires)``
    (``remap_image_sizes``, ml_optimiser.cpp:9046-9055).
    """
    from relax.relion.ctf import fftw_shell_labels
    from relax.relion.optics_groups import group_current_size

    shells = fftw_shell_labels(int(box_size), int(current_size), centered_rows=False)
    sigma2 = np.atleast_2d(np.asarray(sigma2_noise, dtype=np.float64))
    shells = shells[(shells > 0) & (shells < sigma2.shape[1])]
    logsigma2 = np.log(2.0 * np.pi * sigma2[:, shells]).sum(axis=1)
    for group, (box, scale, remap) in (group_grids or {}).items():
        if int(box) == int(box_size) and float(scale) == 1.0 and float(remap) == 1.0:
            continue  # the model's own grid: the shared sum above
        ires = fftw_shell_labels(int(box), group_current_size(current_size, box, scale), centered_rows=False)
        ires = ires[ires > 0]
        # RELION's ROUND of a positive value: (int)(x + 0.5).
        remapped = np.floor(float(remap) * ires + 0.5).astype(np.int64)
        remapped = remapped[remapped < sigma2.shape[1]]
        logsigma2[group] = np.log(2.0 * np.pi * sigma2[group, remapped]).sum()
    return np.asarray(log_evidence, np.float64) - np.asarray(n_images, np.float64) * logsigma2[np.asarray(groups, np.int64)]


def relion_firstiter_cc_log_likelihood_contributions(best_log_score, n_images):
    """RELION's dLL of a cross-correlation iteration (``--firstiter_cc`` iteration 1, ``--always_cc``):
    ``-min_diff2 / nr_images`` (acc_ml_optimiser_impl.h:3571-3572), ``best_log_score`` being ``-min_diff2``."""

    return np.asarray(best_log_score, np.float64) / np.asarray(n_images, np.float64)
