"""Per-half noise initialization, sufficient statistics and posterior updates.

Keep shell profiles, expanded pixel variances and their aliasing explicit.
The controller and replay diagnostics import this owner directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from relax.helpers.types import make_noise_stats, total_sumw

logger = logging.getLogger(__name__)


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
    """The one remaining mode decision of the ``sigma_offset`` update.

    Reads from ``per_half`` (the iteration's ``PerHalfOutputs``): ``noise_stats`` and, for Class3D,
    ``noise_stats_per_class``; from ``sigma_offset`` (the current ``SigmaOffset``): ``per_half_angstrom``.

    Class3D shares one pooled value between the halves and reports per-class
    values (``update_class_sigma_offset_from_posterior``); K=1 updates each
    half-model on its own (``update_k1_sigma_offset_from_posterior``). Remove
    this dispatch when the K1 and Class3D trajectories call those directly.
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


def _normalize_noise_variance_per_half(init_noise_variance):
    """Return a list of the two halves' flattened noise-variance arrays.

    RELION stores and updates ``sigma2_noise`` separately for each half-model.
    Legacy RECOVAR callers pass one shared image-shaped array; keep that path
    by duplicating the shared vector.
    """
    n_halves = 2
    if isinstance(init_noise_variance, (list, tuple)):
        if len(init_noise_variance) != n_halves:
            raise ValueError(
                f"Expected {n_halves} per-half noise arrays, got {len(init_noise_variance)}",
            )
        per_half = [_flat_noise_rows(noise_k) for noise_k in init_noise_variance]
    else:
        noise_arr = jnp.asarray(init_noise_variance)
        if noise_arr.ndim == 1:
            shared = noise_arr.reshape(-1)
            per_half = [jnp.array(shared) for _ in range(n_halves)]
        elif noise_arr.ndim == 2 and noise_arr.shape[0] == n_halves:
            per_half = [jnp.asarray(noise_arr[k]).reshape(-1) for k in range(n_halves)]
        else:
            raise ValueError(
                "init_noise_variance must be a flat shared array or a "
                f"({n_halves}, image_size) per-half array; got shape {tuple(noise_arr.shape)}",
            )

    sizes = [tuple(noise_k.shape) for noise_k in per_half]
    if len(set(sizes)) != 1:
        raise ValueError(f"Per-half noise arrays must have the same shape; got {sizes}")
    return per_half


def _flat_noise_rows(noise_k):
    """One half's noise: a flat image vector, or ``[G, P]`` rows for G > 1 optics groups.

    RELION keeps one ``sigma2_noise`` spectrum per optics group
    (``MlModel::sigma2_noise[optics_group]``). One group keeps the flat vector,
    so single-group runs see exactly the array they saw before.
    """
    noise_k = jnp.asarray(noise_k)
    if noise_k.ndim == 2 and noise_k.shape[0] > 1:
        return noise_k.reshape(noise_k.shape[0], -1)
    return noise_k.reshape(-1)



def _mean_noise_variance(noise_variance_per_half):
    """Average per-half image noise for diagnostics and compatibility outputs."""
    return jnp.mean(
        jnp.stack([_flat_noise_rows(noise_k) for noise_k in noise_variance_per_half], axis=0),
        axis=0,
    )



def _noise_radial_history(noise_variance_per_half, image_shape, *, dtype):
    """Build fresh per-half shell profiles and their mean in the requested dtype.

    Pixel-array normalization and noise estimation remain with the caller.
    Preserve the float64 host shell reduction before the final JAX cast.
    """
    from relax.relion.relion_metadata import _radial_profile_from_noise_variance

    def radial(noise_k):
        noise_k = _flat_noise_rows(noise_k)
        if noise_k.ndim == 1:
            return _radial_profile_from_noise_variance(noise_k, image_shape)
        return np.stack([_radial_profile_from_noise_variance(row, image_shape) for row in noise_k])

    per_half = [radial(noise_k) for noise_k in noise_variance_per_half]
    mean = jnp.asarray(np.mean(np.stack(per_half, axis=0), axis=0), dtype=dtype)
    return per_half, mean



def _combined_noise_stats(noise_stats_per_half):
    """Sum half-set noise sufficient statistics before RELION Class3D normalization.

    Both halves carry statistics (the caller runs ``_require_noise_stats_of_both_halves`` first). A
    half without mass adds exact zeros, and with several optics groups its placeholder sums need not
    have the scored half's ``[G, n]`` layout, so it is left out. ``sumw`` keeps the per-group ``[G]``
    layout when the sums have one.
    """

    stats = list(noise_stats_per_half)
    with_mass = [stats_k for stats_k in stats if total_sumw(stats_k.sumw) > 0.0]
    if with_mass:
        stats = with_mass
    wsum_sigma2_noise = np.sum(
        [np.asarray(stats_k.wsum_sigma2_noise, dtype=np.float64) for stats_k in stats],
        axis=0,
    )
    wsum_img_power = np.sum(
        [np.asarray(stats_k.wsum_img_power, dtype=np.float64) for stats_k in stats],
        axis=0,
    )
    wsum_sigma2_offset = float(sum(float(stats_k.wsum_sigma2_offset) for stats_k in stats))
    sumw = sum(np.asarray(stats_k.sumw, dtype=np.float64) for stats_k in stats)
    sumw = float(sumw) if np.ndim(sumw) == 0 else np.asarray(sumw, dtype=np.float64)

    def _sum_optional_field(name: str, like):
        values = [getattr(stats_k, name, None) for stats_k in stats]
        if all(value is None for value in values):
            return None
        return np.sum(
            [
                np.zeros_like(like, dtype=np.float64) if value is None else np.asarray(value, dtype=np.float64)
                for value in values
            ],
            axis=0,
        )

    wsum_noise_a2 = _sum_optional_field("wsum_noise_a2", wsum_sigma2_noise)
    wsum_noise_xa = _sum_optional_field("wsum_noise_xa", wsum_sigma2_noise)
    return make_noise_stats(
        wsum_sigma2_noise=wsum_sigma2_noise,
        wsum_img_power=wsum_img_power,
        wsum_sigma2_offset=wsum_sigma2_offset,
        sumw=sumw,
        wsum_noise_a2=wsum_noise_a2,
        wsum_noise_xa=wsum_noise_xa,
        array_dtype=jnp.float64,
    )



def _per_optics_group_sigma2_noise(
    stats,
    previous_radial,
    previous_rows,
    image_shape,
    *,
    ctf_premultiplied=False,
    summed_current_size=None,
    nyquist_column_counting="relion",
):
    """One half's M-step noise update with one spectrum per optics group.

    ``stats`` carries ``[G, n_shells]`` sums and ``[G]`` ``sumw`` (RELION's
    ``wsum_model.sigma2_noise[igroup]`` and ``sumw_group[igroup]``). Each group is
    normalised on its own, as ``maximizationOtherParameters`` does
    (``ml_optimiser.cpp:5246-5285``); a group without noise sums keeps its previous
    spectrum. Returns the ``[G, n_shells]`` shell profiles and ``[G, P]`` pixel rows.
    """

    from relax.reconstruction import noise_relion

    wsum = np.asarray(stats.wsum_sigma2_noise, dtype=np.float64)
    power = np.asarray(stats.wsum_img_power, dtype=np.float64)
    sumw = np.asarray(stats.sumw, dtype=np.float64).reshape(-1)
    n_groups = wsum.shape[0]
    if power.shape != wsum.shape or sumw.shape != (n_groups,) or previous_radial.shape != wsum.shape:
        raise ValueError(
            f"per-group noise sums disagree: wsum {wsum.shape}, img power {power.shape}, "
            f"sumw {sumw.shape}, previous {previous_radial.shape}"
        )
    radial = previous_radial.copy()
    rows = [jnp.asarray(row) for row in jnp.asarray(previous_rows).reshape(n_groups, -1)]
    for g in range(n_groups):
        if np.sum(wsum[g] + power[g]) == 0.0:
            continue
        radial[g] = np.asarray(
            noise_relion.normalize_wsum_to_sigma2_noise(
                wsum[g],
                power[g],
                sumw[g],
                image_shape,
                ctf_premultiplied=ctf_premultiplied,
                summed_current_size=summed_current_size,
                nyquist_column_counting=nyquist_column_counting,
            ),
            dtype=np.float64,
        )
        rows[g] = _shell_profile_pixel_row(radial[g], image_shape)
    return radial, jnp.stack(rows)


@dataclass(frozen=True)
class NoiseModel:
    """Two half-model spectra and their pixel expansions.

    Each half has flat pixel/shell arrays, or ``(G, P)`` / ``(G, S)`` arrays
    for several optics groups. Cached averages serve diagnostics; their dtype
    depends on initialization, replay or estimation and is retained separately.
    K1 estimation replaces entries in the pixel list; Class3D shares its pixel
    array between halves and keeps independent host shell profiles.
    """

    variance_per_half: list
    radial_per_half: list
    average_variance: object
    average_radial: object


def initialize_noise_model(variance_per_half, *, average_variance, image_shape, dtype):
    """Derive shell profiles after the caller's pixel normalization and averaging."""
    radial_per_half, average_radial = _noise_radial_history(
        variance_per_half, image_shape, dtype=dtype,
    )
    return NoiseModel(variance_per_half, radial_per_half, average_variance, average_radial)


def noise_model_from_pixels(noise_variance, image_shape, *, dtype):
    """Normalize input/replay pixel noise and derive its diagnostic shell profiles."""
    variance_per_half = _normalize_noise_variance_per_half(noise_variance)
    average_variance = _mean_noise_variance(variance_per_half)
    return initialize_noise_model(
        variance_per_half, average_variance=average_variance, image_shape=image_shape, dtype=dtype,
    )


def _shell_profile_pixel_row(shell_profile, image_shape):
    """Expand one ``(S,)`` shell profile to its flat pixel row; the operand keeps its array type and dtype."""
    from recovar.reconstruction import noise

    return jnp.asarray(noise.make_radial_noise(shell_profile, image_shape)).reshape(-1)


def noise_pixel_rows(noise_shells, image_shape):
    """Expand one half's noise shells (``(S,)`` or ``(G, S)``) to pixel rows."""
    shells = np.asarray(noise_shells, dtype=np.float64)
    if shells.ndim == 1:
        return _shell_profile_pixel_row(shells, image_shape)
    return jnp.stack([_shell_profile_pixel_row(row, image_shape) for row in shells])


def noise_model_from_shells(noise_shells, image_shape):
    """Restore checkpoint spectra without the initialization scoring-dtype cast."""
    variance_per_half = [noise_pixel_rows(shells, image_shape) for shells in noise_shells]
    average_variance = _mean_noise_variance(variance_per_half)
    radial_per_half = [np.asarray(shells, dtype=np.float64) for shells in noise_shells]
    average_radial = jnp.asarray(np.mean(np.stack(radial_per_half, axis=0), axis=0))
    return NoiseModel(variance_per_half, radial_per_half, average_variance, average_radial)


@dataclass
class NoiseUpdateResult:
    """Updated model and float64 shell estimates for iteration history.

    CC retains the previous model's shell objects, whose precision can differ
    from the history estimates. Other updates share those estimates with the
    next model's radial profiles.
    """

    model: NoiseModel
    noise_from_res: np.ndarray
    noise_from_res_per_half: list


def datasets_store_premultiplied_ctf(experiment_datasets) -> bool:
    """Whether some optics group stores CTF-premultiplied images (RELION's ``hasCtfPremultiplied``).

    A subtomogram half is asked through its flat per-tilt dataset (``half.images``).
    """
    from relax.refinement.tomo_half import TomoHalf
    from relax.relion import relion_ctf

    return any(
        relion_ctf.dataset_has_premultiplied_ctf(
            dataset.images if isinstance(dataset, TomoHalf) else dataset, tuple(int(v) for v in dataset.image_shape)
        )
        for dataset in experiment_datasets
    )


def _require_noise_stats_of_both_halves(noise_stats_per_half):
    if noise_stats_per_half[0] is None or noise_stats_per_half[1] is None:
        raise RuntimeError(
            "RELION mode expected per-half NoiseStats from the EM engine; "
            "ensure accumulate_noise=True is plumbed through pass 2.",
        )


def _noise_update_keeping_previous_spectra(model: NoiseModel) -> NoiseUpdateResult:
    """RELION's iter-1 CC emulation skips the first-iteration noise update: the model keeps its arrays."""
    noise_variance_per_half = model.variance_per_half
    previous_noise_radial_per_half = model.radial_per_half
    noise_from_res_per_half = [
        np.asarray(noise_k, dtype=np.float64)
        for noise_k in previous_noise_radial_per_half
    ]
    noise_from_res = np.mean(np.stack(noise_from_res_per_half, axis=0), axis=0)
    logger.info(
        "RELION iter-1 CC emulation: keeping previous sigma2_noise (skip first-iter noise update)",
    )
    return NoiseUpdateResult(
        model=NoiseModel(
            noise_variance_per_half,
            previous_noise_radial_per_half,
            _mean_noise_variance(noise_variance_per_half),
            model.average_radial,
        ),
        noise_from_res=noise_from_res,
        noise_from_res_per_half=noise_from_res_per_half,
    )


def _one_group_sigma2_noise(stats, image_shape, *, ctf_premultiplied, summed_current_size, nyquist_column_counting):
    """One optics group's M-step noise update: the float64 shell profile and its flat pixel row."""
    from relax.reconstruction import noise_relion

    sigma2_noise = noise_relion.normalize_wsum_to_sigma2_noise(
        np.asarray(stats.wsum_sigma2_noise, dtype=np.float64),
        np.asarray(stats.wsum_img_power, dtype=np.float64),
        stats.sumw,
        image_shape,
        ctf_premultiplied=ctf_premultiplied,
        summed_current_size=summed_current_size,
        nyquist_column_counting=nyquist_column_counting,
    )
    return np.asarray(sigma2_noise, dtype=np.float64), _shell_profile_pixel_row(sigma2_noise, image_shape)


def _noise_update_result(
    model: NoiseModel,
    noise_variance_per_half,
    noise_from_res_per_half,
    noise_from_res,
) -> NoiseUpdateResult:
    """Log the new spectra against ``model``'s and build the updated model. The controller hands the update's
    terms to its observer (``RunObserver.noise_updated``)."""
    # Log per-shell noise comparison (first 10 shells) for convergence diagnostics.
    old_noise_radial = np.asarray(model.average_radial).reshape(
        -1, np.shape(noise_from_res)[-1]
    )[0]
    new_noise_radial = np.asarray(noise_from_res).reshape(-1, np.shape(noise_from_res)[-1])[0]
    n_log = min(10, len(new_noise_radial), len(old_noise_radial))
    logger.info(
        "Noise update per shell (first %d, optics group 1): old=[%s] new=[%s]",
        n_log,
        ", ".join(f"{float(x):.3e}" for x in old_noise_radial[:n_log]),
        ", ".join(f"{float(x):.3e}" for x in new_noise_radial[:n_log]),
    )
    new_previous_noise_radial = jnp.asarray(noise_from_res)
    noise_variance = _mean_noise_variance(noise_variance_per_half)
    return NoiseUpdateResult(
        model=NoiseModel(
            noise_variance_per_half,
            noise_from_res_per_half,
            noise_variance,
            new_previous_noise_radial,
        ),
        noise_from_res=noise_from_res,
        noise_from_res_per_half=noise_from_res_per_half,
    )


def update_k1_posterior_noise_variance(
    noise_stats_per_half,
    model: NoiseModel,
    image_shape,
    *,
    firstiter_cc: bool,
    ctf_premultiplied: bool = False,
    summed_current_size=None,
    nyquist_column_counting="relion",
) -> NoiseUpdateResult:
    """RELION-style posterior-weighted noise update of the two half-models.

    Each half normalizes its own ``wsum_sigma2_noise``/``wsum_img_power``
    accumulators via RELION's M-step formula and keeps an independent
    sigma2_noise; the pixel rows replace the entries of the model's list.

    ``ctf_premultiplied`` (some optics group stores CTF-premultiplied images) applies RELION's
    1e-15 sigma2 floor (ml_optimiser.cpp:5273-5274). When ``firstiter_cc`` is true, keeps the previous
    sigma2_noise (matching RELION's iter-1 CC emulation, which skips the
    first-iter noise update).

    ``summed_current_size`` (the expectation's image current size) divides each
    shell by the pixels that expectation summed instead of RELION's full-image
    ``Npix_per_shell`` (``normalize_wsum_to_sigma2_noise``); None is RELION's count.
    ``nyquist_column_counting`` must be the rule the expectation's sums were accumulated with:
    ``"once"`` counts each Hermitian pair of the full-size Nyquist column once.
    """

    _require_noise_stats_of_both_halves(noise_stats_per_half)
    if firstiter_cc:
        return _noise_update_keeping_previous_spectra(model)

    noise_variance_per_half = model.variance_per_half
    several_optics_groups = np.ndim(noise_stats_per_half[0].wsum_sigma2_noise) == 2
    noise_from_res_per_half = []
    for k_noise, stats_k in enumerate(noise_stats_per_half):
        if several_optics_groups:
            noise_k, noise_rows_k = _per_optics_group_sigma2_noise(
                stats_k,
                np.asarray(
                    model.radial_per_half[k_noise], dtype=np.float64
                ),
                noise_variance_per_half[k_noise],
                image_shape,
                ctf_premultiplied=ctf_premultiplied,
                summed_current_size=summed_current_size,
                nyquist_column_counting=nyquist_column_counting,
            )
        else:
            noise_k, noise_rows_k = _one_group_sigma2_noise(
                stats_k,
                image_shape,
                ctf_premultiplied=ctf_premultiplied,
                summed_current_size=summed_current_size,
                nyquist_column_counting=nyquist_column_counting,
            )
        noise_from_res_per_half.append(noise_k)
        noise_variance_per_half[k_noise] = noise_rows_k
    noise_from_res = np.mean(np.stack(noise_from_res_per_half, axis=0), axis=0)
    return _noise_update_result(model, noise_variance_per_half, noise_from_res_per_half, noise_from_res)


def update_class_posterior_noise_variance(
    noise_stats_per_half,
    model: NoiseModel,
    image_shape,
    *,
    firstiter_cc: bool,
    ctf_premultiplied: bool = False,
    summed_current_size=None,
    nyquist_column_counting="relion",
) -> NoiseUpdateResult:
    """RELION-style posterior-weighted noise update shared by every class (Class3D ordering).

    Sums the ``wsum_sigma2_noise``/``wsum_img_power`` accumulators from
    both half-sets and normalizes via RELION's M-step formula. Both halves
    take the one pixel array in a new list and independent copies of the
    shell profile.

    ``ctf_premultiplied`` (some optics group stores CTF-premultiplied images) applies RELION's
    1e-15 sigma2 floor (ml_optimiser.cpp:5273-5274). When ``firstiter_cc`` is true, keeps the previous
    sigma2_noise (matching RELION's iter-1 CC emulation, which skips the
    first-iter noise update).

    ``summed_current_size`` (the expectation's image current size) divides each
    shell by the pixels that expectation summed instead of RELION's full-image
    ``Npix_per_shell`` (``normalize_wsum_to_sigma2_noise``); None is RELION's count.
    ``nyquist_column_counting`` must be the rule the expectation's sums were accumulated with:
    ``"once"`` counts each Hermitian pair of the full-size Nyquist column once.
    """

    _require_noise_stats_of_both_halves(noise_stats_per_half)
    if firstiter_cc:
        return _noise_update_keeping_previous_spectra(model)

    combined_noise_stats = _combined_noise_stats(noise_stats_per_half)
    if np.ndim(combined_noise_stats.wsum_sigma2_noise) == 2:
        # Several optics groups (subtomogram Class3D): one spectrum per group, as for K=1.
        noise_from_res, noise_rows = _per_optics_group_sigma2_noise(
            combined_noise_stats,
            np.asarray(model.radial_per_half[0], dtype=np.float64),
            model.variance_per_half[0],
            image_shape,
            ctf_premultiplied=ctf_premultiplied,
            summed_current_size=summed_current_size,
            nyquist_column_counting=nyquist_column_counting,
        )
    else:
        noise_from_res, noise_rows = _one_group_sigma2_noise(
            combined_noise_stats, image_shape,
            ctf_premultiplied=ctf_premultiplied, summed_current_size=summed_current_size,
            nyquist_column_counting=nyquist_column_counting,
        )
    noise_from_res_per_half = [noise_from_res.copy(), noise_from_res.copy()]
    noise_variance_per_half = [noise_rows, noise_rows]
    return _noise_update_result(model, noise_variance_per_half, noise_from_res_per_half, noise_from_res)


def update_posterior_noise_variance(
    noise_stats_per_half,
    model: NoiseModel,
    image_shape,
    *,
    k_class_enabled: bool,
    firstiter_cc: bool,
    ctf_premultiplied: bool = False,
    summed_current_size=None,
    nyquist_column_counting="relion",
) -> NoiseUpdateResult:
    """The one remaining mode decision of the noise update.

    K-class refinement shares one sigma2_noise across classes
    (``update_class_posterior_noise_variance``); K=1 keeps independent
    per-half sigma2_noise (``update_k1_posterior_noise_variance``). Remove
    this dispatch when the K1 and Class3D trajectories call those directly.
    """

    update = update_class_posterior_noise_variance if k_class_enabled else update_k1_posterior_noise_variance
    return update(
        noise_stats_per_half, model, image_shape,
        firstiter_cc=firstiter_cc, ctf_premultiplied=ctf_premultiplied,
        summed_current_size=summed_current_size, nyquist_column_counting=nyquist_column_counting,
    )
