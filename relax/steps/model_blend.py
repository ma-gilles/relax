"""RELION's stochastic-gradient blend of the model's other parameters, shared by VDAM and the PPCA initial model.

``MlOptimiser::maximizationOtherParameters`` updates the class and direction priors, the offset variance and the
noise spectra as ``x *= mu; x += (1 - mu) * wsum / weight`` (ml_optimiser.cpp, the ``(1. - my_mu)`` lines from
``pdf_class`` to ``sigma2_noise``). The callers choose ``mu`` (0 for a full-data update), the units and the
weight; floors and validation stay with them.
"""

from __future__ import annotations

# RELION's lower bound of the offset variance in Angstrom squared (VDAM's sigma2_offset; PPCA's in pixels squared is
# this over the pixel size squared).
MIN_SIGMA2_OFFSET_ANGSTROM2: float = 2.0


def relion_blend(previous, wsum, weight, mu):
    """``previous * mu + (1 - mu) * wsum / weight`` in RELION's grouping, ``((1 - mu) * wsum) / weight``.

    Scalars or arrays (elementwise, NumPy promotion); ``weight`` is the normalization of the weighted sum
    ``wsum``, so ``wsum / weight`` is the measured value that the blend moves ``previous`` towards.
    """
    return previous * mu + (1.0 - mu) * wsum / weight


def blend_offset_variance(previous: float, wsum, sumw, dims: int, mu: float, floor: float) -> float:
    """RELION's ``sigma2_offset`` update: :func:`relion_blend` with weight ``dims * sumw``, at least ``floor``.

    ``wsum`` is the posterior-weighted sum of squared offsets and ``sumw`` its weight; ``dims`` is 2 for in-plane
    particle shifts and 3 for subtomogram shifts. ``previous``, ``floor`` and ``wsum / sumw`` share the caller's
    units (Angstrom squared for VDAM, pixels squared for PPCA).
    """
    return max(float(relion_blend(previous, wsum, dims * sumw, mu)), floor)
