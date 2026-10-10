"""The translation prior width (RELION's sigma2_offset, in Angstrom): the record a run carries
(``SigmaOffset``) and its update from the posterior-weighted offsets of an expectation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from relax.helpers.types import total_sumw

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SigmaOffset:
    """The translation prior width in Angstrom: the value the halves share and each half's own.

    A producer supplies both; the shared value is not always the mean of the two. Without a pair, both halves
    have the shared width.
    """

    shared_angstrom: float
    per_half_angstrom: tuple[float, float] | None = None

    def __post_init__(self):
        shared = float(self.shared_angstrom)
        pair = (shared, shared) if self.per_half_angstrom is None else self.per_half_angstrom
        arr = np.asarray(pair, dtype=np.float64).reshape(-1)
        if arr.size != 2:
            raise ValueError(
                f"translation_sigma_angstrom_per_half must contain exactly two values; got shape {np.asarray(pair).shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("translation_sigma_angstrom_per_half must be finite")
        object.__setattr__(self, "shared_angstrom", shared)
        object.__setattr__(self, "per_half_angstrom", (float(arr[0]), float(arr[1])))

    @classmethod
    def from_halves(cls, widths) -> "SigmaOffset":
        """Each half's width (a pair, or one width for both) with their mean as the shared value."""

        arr = np.asarray(widths, dtype=np.float64).reshape(-1)
        if arr.size == 1:
            arr = np.repeat(arr, 2)
        if arr.size != 2:
            raise ValueError(
                f"translation_sigma_angstrom_per_half must contain exactly two values; got shape {np.asarray(widths).shape}"
            )
        return cls(float(0.5 * (float(arr[0]) + float(arr[1]))), arr)

    def for_half(self, half_index: int) -> float:
        return self.per_half_angstrom[half_index]


@dataclass
class SigmaOffsetUpdateResult:
    """Posterior-weighted ``sigma_offset`` update result.

    RELION gold-standard refinement has one model per half-set, so each half
    updates and consumes its own sigma offset. The scalar value remains the
    mean of the pair for backward-compatible telemetry only.
    """

    current_sigma_offset_angstrom: float
    current_sigma_offset_angstrom_per_half: list[float]
    per_class_sigma_offset_angstrom: np.ndarray | None


def _sigma_offset_from_moment(
    wsum: float,
    sumw: float,
    *,
    current_sigma_offset_angstrom: float,
    state_fallback_offsets_angstrom: float,
    offset_dims: int = 2,
) -> float:
    min_sigma2_angstrom2 = 2.0
    # Any posterior mass carries a moment, zero when every offset sits on its prior centre; RELION then
    # takes the min_sigma2_offset bound (ml_optimiser.cpp:5233-5235).
    if sumw > 0.0:
        return float(np.sqrt(max(wsum / (float(offset_dims) * sumw), min_sigma2_angstrom2)))
    if np.isfinite(state_fallback_offsets_angstrom) and state_fallback_offsets_angstrom > 0.0:
        return max(float(state_fallback_offsets_angstrom), float(np.sqrt(min_sigma2_angstrom2)))
    return float(current_sigma_offset_angstrom)


def _sigma_offset_pair(current_sigma_offset_angstrom_per_half) -> np.ndarray:
    current_per_half = np.asarray(current_sigma_offset_angstrom_per_half, dtype=np.float64).reshape(-1)
    if current_per_half.size != 2 or not np.all(np.isfinite(current_per_half)):
        raise ValueError("current_sigma_offset_angstrom_per_half must contain two finite values")
    return current_per_half


def _half_offset_moment(stats_k) -> tuple[float, float]:
    """One half's ``(wsum_sigma2_offset, sum_weight)``; zeros when the half has no statistics."""
    if stats_k is None:
        return 0.0, 0.0
    wsum_k = float(getattr(stats_k, "wsum_sigma2_offset", 0.0))
    # RELION's sum_weight is the total class mass over all optics groups (ml_optimiser.cpp:5099-5101).
    sumw_k = total_sumw(getattr(stats_k, "sumw", 0.0))
    return wsum_k, sumw_k


def _log_sigma_offset_update(per_half_sigma_offset, current_sigma_offset_angstrom: float) -> None:
    logger.info(
        "C1: sigma_offset updated per half [%.3f, %.3f] Å (mean %.3f Å)",
        per_half_sigma_offset[0],
        per_half_sigma_offset[1],
        current_sigma_offset_angstrom,
    )


def update_k1_sigma_offset_from_posterior(
    *,
    noise_stats_per_half,
    current_sigma_offset_angstrom_per_half,
    state_fallback_offsets_angstrom: float,
    offset_dims: int = 2,
) -> SigmaOffsetUpdateResult:
    """RELION posterior-weighted ``sigma_offset`` update of each half-model.

    Prefer RELION's posterior-weighted sufficient statistic:

        sigma2_offset_new = wsum_sigma2_offset / (offset_dims * sum_weight)

    with ``offset_dims`` 2 for single-particle data and 3 for subtomograms
    (ml_optimiser.cpp:5222-5227). A half without a propagated posterior moment
    uses the hard-assignment fallback independently; pooling the other half's
    posterior into it would not match RELION's gold-standard models.
    """

    current_per_half = _sigma_offset_pair(current_sigma_offset_angstrom_per_half)
    per_half_values = []
    for half_idx, stats_k in enumerate(noise_stats_per_half):
        wsum_k, sumw_k = _half_offset_moment(stats_k)
        per_half_values.append(
            _sigma_offset_from_moment(
                wsum_k,
                sumw_k,
                current_sigma_offset_angstrom=float(current_per_half[half_idx]),
                state_fallback_offsets_angstrom=state_fallback_offsets_angstrom,
                offset_dims=offset_dims,
            )
        )
    per_half_sigma_offset = np.asarray(per_half_values, dtype=np.float64)
    current_sigma_offset_angstrom = float(np.mean(per_half_sigma_offset))
    _log_sigma_offset_update(per_half_sigma_offset, current_sigma_offset_angstrom)
    return SigmaOffsetUpdateResult(
        current_sigma_offset_angstrom=current_sigma_offset_angstrom,
        current_sigma_offset_angstrom_per_half=per_half_sigma_offset.tolist(),
        per_class_sigma_offset_angstrom=None,
    )


def _per_class_sigma_offset_report(
    noise_stats_per_half_per_class, n_classes: int, current_sigma_offset_angstrom: float, offset_dims: int
) -> np.ndarray:
    # D.2: per-class sigma_offset diagnostic. RELION Class3D maintains one
    # shared sigma2_offset in model_general; per-class values here are logged
    # only to help diagnose skewed class posteriors without changing the live
    # shared translation prior.
    per_class_w = np.zeros(n_classes, dtype=np.float64)
    per_class_n = np.zeros(n_classes, dtype=np.float64)
    for half_per_class in noise_stats_per_half_per_class:
        if half_per_class is None:
            continue
        for c, stats_c in enumerate(half_per_class):
            if stats_c is None:
                continue
            per_class_w[c] += float(getattr(stats_c, "wsum_sigma2_offset", 0.0))
            per_class_n[c] += total_sumw(getattr(stats_c, "sumw", 0.0))
    min_sigma2 = 2.0
    per_class_sigma_offset = np.full(n_classes, current_sigma_offset_angstrom, dtype=np.float64)
    for c in range(n_classes):
        if per_class_w[c] > 0.0 and per_class_n[c] > 0.0:
            s2 = max(per_class_w[c] / (float(offset_dims) * per_class_n[c]), min_sigma2)
            per_class_sigma_offset[c] = float(np.sqrt(s2))
    logger.info(
        "C1: per-class sigma_offset = [%s] (cross-class aggregate %.3f Å)",
        ", ".join(f"{s:.3f}" for s in per_class_sigma_offset),
        current_sigma_offset_angstrom,
    )
    return per_class_sigma_offset


def update_class_sigma_offset_from_posterior(
    *,
    noise_stats_per_half,
    noise_stats_per_half_per_class,
    current_sigma_offset_angstrom_per_half,
    n_classes: int,
    state_fallback_offsets_angstrom: float,
    offset_dims: int = 2,
) -> SigmaOffsetUpdateResult:
    """RELION Class3D posterior-weighted ``sigma_offset`` update: one value for both halves.

    The halves' posterior moments are pooled into

        sigma2_offset_new = wsum_sigma2_offset / (offset_dims * sum_weight)

    with ``offset_dims`` 2 for single-particle data and 3 for subtomograms
    (ml_optimiser.cpp:5222-5227); without a pooled moment the hard-assignment
    fallback applies to the mean of the current pair. The per-class values of
    ``noise_stats_per_half_per_class`` are a logged diagnostic only.
    """

    current_per_half = _sigma_offset_pair(current_sigma_offset_angstrom_per_half)
    pooled_wsum = 0.0
    pooled_sumw = 0.0
    for stats_k in noise_stats_per_half:
        wsum_k, sumw_k = _half_offset_moment(stats_k)
        pooled_wsum += wsum_k
        pooled_sumw += sumw_k
    shared_sigma_offset = _sigma_offset_from_moment(
        pooled_wsum,
        pooled_sumw,
        current_sigma_offset_angstrom=float(np.mean(current_per_half)),
        state_fallback_offsets_angstrom=state_fallback_offsets_angstrom,
        offset_dims=offset_dims,
    )
    per_half_sigma_offset = np.full(len(noise_stats_per_half), shared_sigma_offset, dtype=np.float64)
    per_class_sigma_offset = _per_class_sigma_offset_report(
        noise_stats_per_half_per_class, n_classes, shared_sigma_offset, offset_dims,
    )
    _log_sigma_offset_update(per_half_sigma_offset, shared_sigma_offset)
    return SigmaOffsetUpdateResult(
        current_sigma_offset_angstrom=shared_sigma_offset,
        current_sigma_offset_angstrom_per_half=per_half_sigma_offset.tolist(),
        per_class_sigma_offset_angstrom=per_class_sigma_offset,
    )


def update_c1_sigma_offset_from_posterior(
    per_half,
    sigma_offset,
    *,
    n_classes: int,
    state_fallback_offsets_angstrom: float,
    offset_dims: int = 2,
) -> SigmaOffsetUpdateResult:
    """The ``sigma_offset`` update of the run's mode.

    Reads from ``per_half`` (the iteration's ``PerHalfOutputs``): ``noise_stats`` and, for Class3D,
    ``noise_stats_per_class``; from ``sigma_offset`` (the current ``SigmaOffset``): ``per_half_angstrom``.

    Class3D shares one pooled value between the halves and reports per-class
    values (``update_class_sigma_offset_from_posterior``); K=1 updates each
    half-model on its own (``update_k1_sigma_offset_from_posterior``).
    """

    if n_classes > 1:
        return update_class_sigma_offset_from_posterior(
            noise_stats_per_half=per_half.noise_stats,
            noise_stats_per_half_per_class=per_half.noise_stats_per_class,
            current_sigma_offset_angstrom_per_half=sigma_offset.per_half_angstrom,
            n_classes=n_classes,
            state_fallback_offsets_angstrom=state_fallback_offsets_angstrom,
            offset_dims=offset_dims,
        )
    return update_k1_sigma_offset_from_posterior(
        noise_stats_per_half=per_half.noise_stats,
        current_sigma_offset_angstrom_per_half=sigma_offset.per_half_angstrom,
        state_fallback_offsets_angstrom=state_fallback_offsets_angstrom,
        offset_dims=offset_dims,
    )
