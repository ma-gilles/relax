"""Reference state, numbered prior estimation and map reconstruction.

Reconstruction, first-CC reporting tapers and their private numerical helpers
share this owner. Controllers install the returned maps and priors explicitly;
final all-data orchestration shares numerical primitives with its own policies.
"""

from __future__ import annotations

import functools
import gc
import logging
import math
from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from recovar.core import fourier_transform_utils, mask

from relax.helpers.orientation_priors import (
    class_weights_from_direction_prior,
)
from relax.helpers.resolution import (
    _firstiter_cc_ini_high_tapered,
    _firstiter_cc_ini_high_tau2_taper,
    shell_index_to_resolution_angstrom,
)
from relax.helpers.timing import Stopwatch
from relax.helpers.xla_memory_reserve import SINGLE_WORKING_SET_LIMIT_SHARE, single_working_set_bytes
from relax.reconstruction import regularization_relion
from relax.refinement.ports import ClassPriorEstimated, RunObserver
from relax.refinement.refinement_options import ReconstructionPrograms
from relax.refinement.tomo_half import TomoHalf
from relax.relion import relion_ctf
from relax.relion.reference_initialization import initial_low_pass_filter_references

logger = logging.getLogger(__name__)

_LARGE_IRFFT_TRANSFORM_SIZE_LIMIT = np.iinfo(np.int32).max


@dataclass
class ReferenceModel:
    """Half/class Fourier maps and the tau2 volumes used by their next expectation.

    Maps stay in the existing flat centered Fourier layout. K1 map slots are
    cleared before reconstruction; Class3D slots share one class stack. Tau2
    retains the RECOVAR frame scale and its shared/per-half scoring policy.
    """

    maps: list
    tau2: object
    tau2_per_half: list


def shared_tau2_per_half(tau2) -> list:
    """Both halves score with the one shared tau2: the pair refers to that array twice."""
    return [tau2, tau2]


def initialize_reference_model(
    half_maps, initial_mean_variance, *, use_per_half_mean_variance, k_class_enabled, dtype, log
):
    """Attach shared/per-half tau2 to the already normalized references; a per-half pair's mean is in the
    scoring ``dtype``."""
    if use_per_half_mean_variance:
        if k_class_enabled:
            raise ValueError("per-half scoring tau2 is supported only for K=1")
        if initial_mean_variance.ndim != 2 or initial_mean_variance.shape[0] != 2:
            raise ValueError("per-half scoring tau2 requires init_mean_variance with leading half axis 2")
        mean_variance_per_half = [
            jnp.asarray(initial_mean_variance[0]),
            jnp.asarray(initial_mean_variance[1]),
        ]
        mean_variance = jnp.asarray(
            0.5 * (mean_variance_per_half[0].astype(jnp.float64) + mean_variance_per_half[1].astype(jnp.float64)),
            dtype=dtype,
        )
        log.info("Initialized exact per-half K=1 tau2 priors")
    else:
        mean_variance = initial_mean_variance
        mean_variance_per_half = shared_tau2_per_half(mean_variance)
    return ReferenceModel(maps=half_maps, tau2=mean_variance, tau2_per_half=mean_variance_per_half)


def reference_model_from_snapshot(snapshot, volume_shape, *, k_class_enabled, dtype):
    """Restore checkpoint references and the continuation's shared tau2 layout."""
    from relax.refinement.iteration_snapshot import tau2_mean_variance

    maps = [jnp.asarray(mean) for mean in snapshot.means]
    if k_class_enabled:
        maps[1] = maps[0]
    tau2 = tau2_mean_variance(snapshot, volume_shape, dtype=dtype)
    return ReferenceModel(maps=maps, tau2=tau2, tau2_per_half=shared_tau2_per_half(tau2))


class HostTau2(NamedTuple):
    shared: object
    per_half: list


def _host_tau2_volumes(mean_variance, mean_variance_per_half):
    """Move the K1 tau2 volumes that are read again to host, copying each distinct device array once.

    The per-half reconstruction volumes are not among them: nothing reads them
    after the reconstruction unless per-half scoring is on, and then they are
    the entries of ``mean_variance_per_half``.
    """
    host = {}

    def to_host(value):
        if value is None or isinstance(value, np.ndarray):
            return value
        key = id(value)
        if key not in host:
            host[key] = np.asarray(jax.device_get(value))
        return host[key]

    return HostTau2(
        to_host(mean_variance),
        [to_host(value) for value in mean_variance_per_half],
    )


def _updated_mean_variance_per_half(
    shared_mean_variance,
    updated_mean_variance_per_half,
    *,
    use_per_half_mean_variance,
):
    """Keep historical K=1 scoring on shared tau2 unless explicitly enabled."""

    if use_per_half_mean_variance:
        if len(updated_mean_variance_per_half) != 2:
            raise ValueError("per-half scoring tau2 update requires exactly two halves")
        return [
            jnp.asarray(updated_mean_variance_per_half[0]),
            jnp.asarray(updated_mean_variance_per_half[1]),
        ]
    return shared_tau2_per_half(shared_mean_variance)


def _normalize_class_log_priors(n_classes: int, class_log_priors=None) -> np.ndarray:
    """Return normalized log priors for the class axis."""

    if n_classes < 1:
        raise ValueError(f"n_classes must be >= 1, got {n_classes}")
    if class_log_priors is None:
        return np.full(n_classes, -np.log(float(n_classes)), dtype=np.float64)
    log_priors = np.asarray(class_log_priors, dtype=np.float64)
    if log_priors.shape != (n_classes,):
        raise ValueError(f"class_log_priors must have shape ({n_classes},), got {log_priors.shape}")
    if not np.all(np.isfinite(log_priors)):
        raise ValueError("class_log_priors must be finite")
    max_log_prior = float(np.max(log_priors))
    log_norm = max_log_prior + float(np.log(np.sum(np.exp(log_priors - max_log_prior))))
    return log_priors - log_norm


class ClassMixture(NamedTuple):
    """The class log priors scored against and the class weights they come from, both ``[K]`` float64."""

    log_priors: np.ndarray
    weights: np.ndarray


def class_mixture_from_weights(weights: np.ndarray) -> ClassMixture:
    """The mixture whose log priors are the logarithm of ``weights``."""

    return ClassMixture(np.log(weights), weights)


def _initialize_class_log_priors(n_classes: int, init_class_log_priors=None, init_direction_prior=None) -> ClassMixture:
    """Return normalized log priors for the class axis and class weights, defaulting to uniform."""
    class_log_priors = _normalize_class_log_priors(n_classes, init_class_log_priors)
    class_weights = np.exp(class_log_priors)
    if n_classes > 1 and init_class_log_priors is None and init_direction_prior is not None:
        inferred_class_weights = class_weights_from_direction_prior(init_direction_prior, n_classes)
        if inferred_class_weights is not None:
            if np.any(inferred_class_weights <= 0.0):
                raise ValueError("RELION direction-prior row sums imply a zero-probability class")
            class_weights = inferred_class_weights
            class_log_priors = np.log(class_weights)
    return ClassMixture(class_log_priors, class_weights)


def _snapshot_and_release_previous_k1_means(means):
    """Copy both K1 references to host before releasing their active buffers.

    One host copy per map: a device map's transfer is that copy (a read-only array); a host map is copied.
    """
    previous_means = [
        None if mean is None else mean.copy() if isinstance(mean, np.ndarray) else np.asarray(mean)
        for mean in means
    ]
    # Dropping the list's references frees the device buffers; there is no
    # cycle for a collection to break (every explicit gc.collect() of the 5k
    # K=1 run found nothing and freed no device memory, job 14503450).
    for half_index in range(2):
        means[half_index] = None
    return previous_means


def _normalize_initial_means(init_volume, n_classes: int):
    """Normalize initial references to the refine loop's half/class layout."""

    def _as_class_array(value):
        arr = jnp.asarray(value)
        if n_classes == 1:
            if arr.ndim == 1:
                return arr
            if arr.ndim == 2 and int(arr.shape[0]) == 1:
                return arr[0]
        else:
            if arr.ndim == 1:
                return jnp.tile(arr[None, :], (n_classes, 1))
            if arr.ndim == 2 and int(arr.shape[0]) == n_classes:
                return arr
        raise ValueError(
            "init_volume must be a flat reference, a per-class reference array, "
            "or a pair of per-half references compatible with n_classes="
            f"{n_classes}; got shape {tuple(arr.shape)}",
        )

    if isinstance(init_volume, (list, tuple)) and len(init_volume) == 2:
        return [_as_class_array(init_volume[0]), _as_class_array(init_volume[1])]

    arr = jnp.asarray(init_volume)
    if n_classes == 1 and arr.ndim == 2 and int(arr.shape[0]) == 2:
        return [arr[0], arr[1]]
    if n_classes > 1 and arr.ndim == 3 and int(arr.shape[0]) == 2 and int(arr.shape[1]) == n_classes:
        return [arr[0], arr[1]]
    shared = _as_class_array(arr)
    # Half-map updates replace list entries; immutable initial buffers can be shared.
    return [shared, shared]


def _class_weights_from_posterior(class_posterior_per_half, n_classes: int, previous_weights: np.ndarray) -> np.ndarray:
    """Normalize class posterior sums across both half-sets."""

    counts = np.zeros(n_classes, dtype=np.float64)
    for posterior in class_posterior_per_half:
        if posterior is not None:
            counts += np.asarray(posterior, dtype=np.float64)
    total = float(np.sum(counts))
    if total <= 0.0:
        return np.asarray(previous_weights, dtype=np.float64)
    weights = np.maximum(counts / total, 1e-12)
    return weights / float(np.sum(weights))


def _previous_resolution_angstrom_for_half_join(
    pixel_resolutions, current_resolution, *, box_size: int, voxel_size: float
):
    """Previous-iteration resolution in Å that caps the low-resolution half join.

    The last recorded shell resolution wins when the history has one; a
    non-positive recorded shell leaves the join uncapped. Without history, a
    finite state resolution is used. Mirrors RELION's
    ``XMIPP_MAX(low_resol_join_halves, 1./mymodel.current_resolution)``.
    """

    if pixel_resolutions:
        previous_shell = pixel_resolutions[-1]
        if previous_shell > 0:
            return shell_index_to_resolution_angstrom(previous_shell, box_size, voxel_size)
        return None
    if np.isfinite(float(current_resolution)):
        return float(current_resolution)
    return None


def join_half_accumulators_at_low_resolution(
    numerators_by_half,
    denominators_by_half,
    *,
    accumulator_volume_shape,
    box_size,
    voxel_size,
    padding_factor,
    low_resolution_angstrom,
    pixel_resolutions,
    current_resolution,
    preserve_inputs=True,
    return_retained_first_numerator=False,
):
    """Apply RELION's ``--low_resol_join_halves`` to K=1 half accumulators before the Wiener solve.

    Averaging the low-resolution shells of the two half accumulators forces
    both half-maps to share their low-frequency content, as in
    ``MlOptimiserMpi::joinTwoHalvesAtLowResolution``; without it the half-map
    FSC drifts below 1 at SNR-poor low shells and current-size growth lags.
    The join radius is capped by the previous iteration's resolution so shells
    beyond the map's actual resolution are never joined. The regular and final
    all-data passes call this with their own accumulators. Returns the four
    joined arrays in input order; refinement state is not mutated.
    """

    previous_resolution_angstrom = _previous_resolution_angstrom_for_half_join(
        pixel_resolutions,
        current_resolution,
        box_size=box_size,
        voxel_size=voxel_size,
    )
    return regularization_relion.join_halves_at_low_resolution(
        numerators_by_half[0],
        numerators_by_half[1],
        denominators_by_half[0],
        denominators_by_half[1],
        accumulator_volume_shape,
        voxel_size,
        box_size,
        low_resolution_angstrom,
        current_resolution_angstrom=previous_resolution_angstrom,
        padding_factor=padding_factor,
        **({"preserve_inputs": False} if not preserve_inputs else {}),
        **({"return_retained_first_numerator": True} if return_retained_first_numerator else {}),
    )


_CLASS_TAU2_DETAIL_KEYS = (
    "prior_shells",
    "sigma2_shells",
    "avg_weight_shells",
    "shell_sum",
    "shell_count",
    "fsc_shells",
    "ssnr_shells",
)


def _class_tau2_from_iref_power_spectrum(
    iref_fourier,
    volume_shape,
    *,
    padding_factor,
    current_size: int,
    frame_scale: float,
    projector_power_spectrum=None,
    shell_pair_counting="relion",
):
    """Class3D tau2 for one class from its previous ``Iref`` power spectrum.

    Dense RECOVAR accumulators live in the historical unnormalised image frame
    (RELION BPref weight = ``Ft_ctf * N^4``), so the RELION-frame tau2 is
    scaled by ``frame_scale`` before the Wiener solve. Returns the
    RECOVAR-frame tau2 volume, the RELION-frame radial shells and the
    RECOVAR-frame radial shells, all in the RELION result dtype.
    ``projector_power_spectrum`` is the class's spectrum from this iteration's
    scoring projector setup, when it has one (see
    ``compute_relion_tau2_from_iref_power_spectrum``).
    """

    mean_signal_variance_relion, details = regularization_relion.compute_relion_tau2_from_iref_power_spectrum(
        iref_fourier,
        volume_shape,
        padding_factor=padding_factor,
        current_size=current_size,
        return_details=True,
        projector_power_spectrum=projector_power_spectrum,
        shell_pair_counting=shell_pair_counting,
    )
    mean_signal_variance = mean_signal_variance_relion * jnp.asarray(
        frame_scale,
        dtype=mean_signal_variance_relion.dtype,
    )
    tau2_shells_relion_frame = jnp.asarray(
        details["tau2_shells"],
        dtype=mean_signal_variance.dtype,
    )
    tau2_shells_recovar_frame = tau2_shells_relion_frame * jnp.asarray(
        frame_scale,
        dtype=mean_signal_variance.dtype,
    )
    return mean_signal_variance, tau2_shells_relion_frame, tau2_shells_recovar_frame


def _class_tau2_update_details(
    Ft_ctf_class,
    tau2_shells_recovar_frame,
    shell_stats,
    settings: ReconstructionSettings,
    *,
    current_size: int,
    full_half_axis,
    accumulator_volume_shape,
    average_ctf2=None,
    shell_pair_counting="relion",
):
    """Data-vs-prior and the host tau2 detail record for one class.

    Reads from ``settings``: ``volume_shape``, ``padding_factor`` and ``tau2_fudge``.

    ``shell_stats`` are the round-shell weight statistics of ``Ft_ctf_class``.
    ``average_ctf2`` is RELION's average CTF^2 of CTF-premultiplied images
    (:func:`relax.relion.relion_ctf.premultiplied_average_ctf2`), or None: RELION
    divides ``invtau2`` by it where it is positive (backprojector.cpp:1277-1279), which
    multiplies ``data_vs_prior`` by it. The record uses the K=1 per-half key layout with
    ``fsc_shells`` set to ``None``. Returns ``(data_vs_prior, details)``.
    """

    data_vs_prior = regularization_relion.compute_data_vs_prior(
        Ft_ctf_class,
        tau2_shells_recovar_frame,
        settings.volume_shape,
        padding_factor=settings.padding_factor,
        tau2_fudge=settings.tau2_fudge,
        current_size=current_size,
        full_half_axis=full_half_axis,
        accumulator_volume_shape=accumulator_volume_shape,
        shell_pair_counting=shell_pair_counting,
    )
    if average_ctf2 is not None:
        count = min(int(data_vs_prior.shape[0]), int(np.asarray(average_ctf2).shape[0]))
        factor = np.ones(int(data_vs_prior.shape[0]))
        factor[:count] = np.where(np.asarray(average_ctf2[:count]) > 0.0, average_ctf2[:count], 1.0)
        data_vs_prior = data_vs_prior * jnp.asarray(factor, dtype=data_vs_prior.dtype)
    details = {
        "prior_shells": np.asarray(tau2_shells_recovar_frame, dtype=np.float64),
        "sigma2_shells": np.asarray(
            jnp.where(
                shell_stats["avg_weight_shells"] > 0,
                1.0 / (settings.padding_factor**3 * shell_stats["avg_weight_shells"]),
                0.0,
            ),
            dtype=np.float64,
        ),
        "avg_weight_shells": np.asarray(shell_stats["avg_weight_shells"], dtype=np.float64),
        "shell_sum": np.asarray(shell_stats["shell_sum"], dtype=np.float64),
        "shell_count": np.asarray(shell_stats["shell_count"], dtype=np.float64),
        "fsc_shells": None,
        "ssnr_shells": np.asarray(data_vs_prior, dtype=np.float64),
    }
    return data_vs_prior, details


def _stack_class_tau2_update_details(details_per_class):
    """Stack per-class tau2 detail records along a leading class axis.

    ``fsc_shells`` stays ``None``: Class3D has no half-set FSC.
    """

    return {
        key: None if key == "fsc_shells" else np.stack([details[key] for details in details_per_class], axis=0)
        for key in _CLASS_TAU2_DETAIL_KEYS
    }


@dataclass(frozen=True)
class ClassPriorEstimate:
    """One class's prior, weight statistics and data-vs-prior estimate.

    ``variance`` and ``shells`` use the RECOVAR frame. ``relion_shells`` keeps
    the unscaled diagnostic spectrum. ``details`` holds the weight statistics on round shells.
    """

    variance: object
    shells: object
    relion_shells: object
    data_vs_prior: object
    details: dict


def estimate_class_prior(
    references,
    denominators,
    *,
    class_index,
    settings: ReconstructionSettings,
    current_size,
    accumulator_shape,
    full_half_axis,
    frame_scale,
    projector_power_spectrum=None,
    replay_tau2_shells=None,
    average_ctf2=None,
) -> ClassPriorEstimate:
    """Estimate a class prior from Iref power or diagnostic replay, then its weights.

    Index inside the operation so reference/projector views are created only
    on the native path and weight views follow prior estimation.
    """
    if replay_tau2_shells is not None:
        shells = jnp.asarray(replay_tau2_shells[class_index], dtype=jnp.float32)
        from recovar import utils

        variance = jnp.asarray(
            utils.make_radial_image(shells, settings.volume_shape, extend_last_frequency=True),
            dtype=jnp.float32,
        ).reshape(-1)
        relion_shells = shells / jnp.asarray(frame_scale, dtype=shells.dtype)
    else:
        variance, relion_shells, shells = _class_tau2_from_iref_power_spectrum(
            references[class_index],
            settings.volume_shape,
            padding_factor=settings.padding_factor,
            current_size=current_size,
            frame_scale=frame_scale,
            # The scoring projector's spectrum counts as RELION does; another counting builds its own.
            projector_power_spectrum=(
                None
                if projector_power_spectrum is None or settings.shell_pair_counting != "relion"
                else projector_power_spectrum[class_index]
            ),
            shell_pair_counting=settings.shell_pair_counting,
        )
    weight_shells = regularization_relion._compute_relion_weight_shell_stats(
        denominators[class_index],
        settings.volume_shape,
        padding_factor=settings.padding_factor,
        r_max=current_size // 2,
        shell_rounding="round",
        full_half_axis=full_half_axis,
        accumulator_volume_shape=accumulator_shape,
        shell_pair_counting=settings.shell_pair_counting,
    )
    data_vs_prior, details = _class_tau2_update_details(
        denominators[class_index],
        shells,
        weight_shells,
        settings,
        current_size=current_size,
        full_half_axis=full_half_axis,
        accumulator_volume_shape=accumulator_shape,
        average_ctf2=average_ctf2,
        shell_pair_counting=settings.shell_pair_counting,
    )
    return ClassPriorEstimate(
        variance=variance,
        shells=shells,
        relion_shells=relion_shells,
        data_vs_prior=data_vs_prior,
        details=details,
    )


@dataclass(frozen=True)
class ClassPriorAggregation:
    """Class-axis priors, scheduling curves and detail rows for ordered publication.

    Variance and shell stacks use the RECOVAR frame. The controller publishes
    the data-vs-prior curve before aggregating detail rows, preserving RELION's
    numbered M-step order. No reference maps or unused class scratch are held.
    """

    variance: object
    shells: object
    data_vs_prior: np.ndarray
    details_per_class: list[dict]


def estimate_class_priors(
    previous_half_maps,
    combined_numerators,
    combined_denominators,
    settings: ReconstructionSettings,
    *,
    half_denominators,
    halves,
    n_classes,
    iteration,
    current_size,
    image_current_size,
    accumulator_shape,
    full_half_axis,
    projector_power_spectrum,
    class_tau2,
    scoring_dtype,
    log,
    observer: RunObserver | None = None,
) -> ClassPriorAggregation:
    """Prepare and aggregate numbered Class3D priors from the previous Iref.

    Own diagnostic replay admission, premultiplied-CTF adaptation, ordered class
    computation/capture and class-axis reductions. The controller installs the
    products and applies first-CC taper only after regularized reconstruction.
    See ``docs/math/relion_refinement_algorithm.md`` for the M-step ordering.
    """
    clock = Stopwatch()
    tau2_update_details_per_class = []
    mean_signal_variance_per_class = []
    mean_signal_variance_shells_per_class = []
    data_vs_prior_per_class = []
    # Dense RECOVAR accumulators live in the historical unnormalised
    # image frame: RELION BPref weight = Ft_ctf * N^4. Equivalently,
    # keep Ft_y/Ft_ctf in RECOVAR frame and scale RELION tau2 by N^4
    # before the Wiener solve. The same frame conversion is documented
    # in docs/math/ab_initio_initial_model_algorithm.md.
    kclass_tau2_frame_scale = float(settings.box_size) ** 4
    # The prior shells an input source supplies (``ports.ClassTau2``), or None: the previous references'.
    kclass_tau2_source = class_tau2.source
    observer = RunObserver() if observer is None else observer
    # CTF-premultiplied images: RELION's average CTF^2 correction of data_vs_prior
    # (setAverageCTF2; Class3D has no split halves and does not fix tau2). It averages over
    # images, so a subtomogram half counts its tilt images, each with its particle's scale.
    ctf2_datasets = [half.dataset for half in halves]
    ctf2_scales = [half.scale_corrections for half in halves]
    if isinstance(ctf2_datasets[0], TomoHalf):
        ctf2_scales = [
            None if scales is None else np.repeat(np.asarray(scales), np.diff(dataset.unit_image_offsets))
            for dataset, scales in zip(ctf2_datasets, ctf2_scales)
        ]
        ctf2_datasets = [dataset.images for dataset in ctf2_datasets]
    average_ctf2 = relion_ctf.premultiplied_average_ctf2(
        ctf2_datasets, ctf2_scales, image_current_size, settings.box_size
    )
    for class_idx in range(n_classes):
        log.info(
            "Class3D tau2 update start: iter=%d class=%d/%d current_size=%d source=%s spectrum=%s",
            iteration + 1,
            class_idx + 1,
            n_classes,
            int(current_size),
            kclass_tau2_source,
            (
                "scoring projector"
                if projector_power_spectrum is not None and settings.shell_pair_counting == "relion"
                else "host transform"
            ),
        )
        class_prior = estimate_class_prior(
            previous_half_maps[0],
            combined_denominators,
            class_index=class_idx,
            settings=settings,
            current_size=current_size,
            accumulator_shape=accumulator_shape,
            full_half_axis=full_half_axis,
            frame_scale=kclass_tau2_frame_scale,
            projector_power_spectrum=projector_power_spectrum,
            replay_tau2_shells=class_tau2.shells,
            average_ctf2=average_ctf2,
        )
        mean_signal_variance_per_class.append(class_prior.variance)
        mean_signal_variance_shells_per_class.append(class_prior.shells)
        data_vs_prior_per_class.append(class_prior.data_vs_prior)
        tau2_update_details_per_class.append(class_prior.details)
        observer.class_prior_estimated(ClassPriorEstimated(
            iteration, class_idx, class_prior, numerators=combined_numerators, denominators=combined_denominators,
            half_denominators=half_denominators, references=previous_half_maps, settings=settings,
            current_size=current_size, source=kclass_tau2_source, accumulator_shape=accumulator_shape,
            full_half_axis=full_half_axis, frame_scale=kclass_tau2_frame_scale,
        ))
        log.info(
            "Class3D tau2 update done: iter=%d class=%d/%d elapsed=%.1fs",
            iteration + 1,
            class_idx + 1,
            n_classes,
            clock.seconds,
        )
    mean_signal_variance = jnp.stack(mean_signal_variance_per_class, axis=0)
    mean_signal_variance_shells = jnp.stack(mean_signal_variance_shells_per_class, axis=0)
    data_vs_prior_iter = np.stack(
        [np.asarray(dvp, dtype=scoring_dtype) for dvp in data_vs_prior_per_class],
        axis=0,
    )
    return ClassPriorAggregation(
        variance=mean_signal_variance,
        shells=mean_signal_variance_shells,
        data_vs_prior=data_vs_prior_iter,
        details_per_class=tau2_update_details_per_class,
    )


def merged_half_map(means):
    """The mean of the two half maps (or half class stacks).

    Class3D's two slots hold one stack, and (x + x) / 2 is x: no second box-scale stack (relax#49: 1.23 GiB at the
    end of a box-380 K3 run on a 16 GB card).
    """
    return means[0] if means[0] is means[1] else (means[0] + means[1]) / 2


def weighted_class_merge(class_means, class_weights):
    """One map from a ``(K, V)`` class stack, weighted by the ``(K,)`` class weights."""
    class_weights_jax = jnp.asarray(class_weights, dtype=class_means.real.dtype)
    return jnp.sum(class_weights_jax[:, None] * class_means, axis=0)


def _stable_reconstruction_class(current_size, vol_shape, padding_factor, accumulator_volume_shape, tau_is_1d):
    """``(physical current size, physical accumulator shape)`` of the reconstruction class, or None.

    With the resident engine's stable Fourier windows on, RELION's
    reconstruction runs in one class for the whole run, the full box: the
    current-size accumulator is zero-padded to the full-box cube and
    ``post_process_from_filter_v2`` takes the box as its static bound and
    RELION's size as the traced ``logical_current_size``, so one program serves
    every iteration (recovar 8633a2a24). The padded voxels lie outside every
    logical support, so the reconstruction is the logical one. The inverse
    transform already runs on the padded full box at every size, so the larger
    accumulator adds only elementwise work, and no early iteration needs more
    memory than the final full-box one.
    """

    from relax.helpers.half_volume_mstep import relion_backprojector_volume_shape

    if current_size is None or accumulator_volume_shape is None or not tau_is_1d:
        return None
    box = int(vol_shape[0])
    logical = int(current_size)
    if logical <= 0 or logical > box:
        return None
    logical_shape = tuple(
        int(v) for v in relion_backprojector_volume_shape(vol_shape, padding_factor, current_size=logical)
    )
    if logical_shape != tuple(int(v) for v in accumulator_volume_shape):
        return None
    physical_shape = tuple(
        int(v) for v in relion_backprojector_volume_shape(vol_shape, padding_factor, current_size=box)
    )
    return box, physical_shape


def _stable_unregularized_class(vol_shape, padding_factor, accumulator_volume_shape, tau, current_size):
    """The full-box accumulator shape an unregularized reconstruction runs in, or None.

    Without a prior and without ``current_size`` (the unregularized half maps
    and class means), the accumulator's own size sets the Wiener mask radius
    and the floor's clamp shell. Zero-padded to the full-box cube with that
    size passed as recovar's traced ``logical_accumulator_size``, every
    iteration's accumulator runs in one program instead of one per size; the
    padded voxels lie outside the logical support.
    """

    from relax.helpers.half_volume_mstep import relion_backprojector_volume_shape

    if tau is not None or current_size is not None or accumulator_volume_shape is None:
        return None
    logical_shape = tuple(int(v) for v in accumulator_volume_shape)
    physical_shape = tuple(int(v) for v in relion_backprojector_volume_shape(vol_shape, padding_factor, current_size=int(vol_shape[0])))
    size = logical_shape[0]
    if len(set(logical_shape)) != 1 or size % 2 == 0 or size >= physical_shape[0]:
        return None
    return physical_shape


@functools.partial(jax.jit, static_argnames=("logical_shape", "physical_shape"))
def _pad_accumulators_to_class(Ft_ctf, Ft_y, logical_shape, physical_shape):
    """Both accumulators zero-padded to the class cube, in one program (eager: four per new size)."""

    return (
        _pad_accumulator_to_class(Ft_ctf, logical_shape, physical_shape),
        _pad_accumulator_to_class(Ft_y, logical_shape, physical_shape),
    )


def _pad_accumulator_to_class(values, logical_shape, physical_shape):
    """Zero-pad a centered full or packed-half (x, y, z>=0) accumulator to a larger odd cube."""

    logical_size, physical_size = int(logical_shape[0]), int(physical_shape[0])
    pad = (physical_size - logical_size) // 2
    values = jnp.asarray(values)
    flat = values.ndim == 1
    full = int(values.size) == logical_size**3
    grid = values.reshape((logical_size,) * 3 if full else (logical_size, logical_size, logical_size // 2 + 1))
    widths = [(pad, pad), (pad, pad), (pad, pad) if full else (0, physical_size // 2 - logical_size // 2)]
    padded = jnp.pad(grid, widths)
    return padded.reshape(-1) if flat else padded


# RECOVAR's name of each gridding-correction window (``post_process_from_filter_v2``).
_RECOVAR_GRIDDING_CORRECT = {"radial": "radial", "separable": "square"}


def _reconstruct_volume_eager(
    Ft_ctf,
    Ft_y,
    vol_shape,
    padding_factor,
    tau,
    tau2_fudge,
    projection_padding_factor,
    use_spherical_mask=True,
    grid_correct=True,
    minres_map=0,
    current_size=None,
    return_real_space=False,
    accumulator_volume_shape=None,
    tau_is_1d=False,
    preserve_output_precision=False,
    relion_filter_scale=None,
    retained_device_numerator=None,
    gridding_kernel="radial",
    *,
    programs: ReconstructionPrograms,
):
    """Eager RELION-style reconstruction from full or half Fourier accumulators.

    ``programs`` (the run's ``ScoringVariants.reconstruction``) chooses the stable full-box class and where the
    padded inverse FFT runs; it changes no value.

    This keeps the reconstruction boundary out of a single monolithic JIT while
    letting the local exact path keep its accumulators in packed half-volume
    layout until the final iDFT boundary. ``gridding_kernel`` is the real-space
    correction window: RELION's ``"radial"`` one or the ``"separable"`` per-axis product.
    relax always leaves ``use_spherical_mask``, ``grid_correct`` and ``return_real_space`` at their
    defaults; the parameters remain for scripts/replay_bpref_contribution_bundle.py and
    scripts/run_k_class_parity.py, which set them.
    """
    if gridding_kernel not in _RECOVAR_GRIDDING_CORRECT:
        raise ValueError(f"gridding_kernel must be 'radial' or 'separable', got {gridding_kernel!r}")
    gridding_correct = _RECOVAR_GRIDDING_CORRECT[gridding_kernel]
    from recovar.reconstruction import relion_functions

    from relax.reconstruction import relion_functions_relion

    # The device CTF row caches yield if the device cannot hand out this reconstruction's largest working set now
    # (the pad into the FFTW half: two halves), before any of its box-scale steps: after a pass, the cached rows left
    # a 16 GB pool too fragmented for the 0.56-1.64 GiB pack and pad requests at box 380 and 448 (relax#49).
    padded_shape = relion_functions._relion_reconstruction_padded_shape(vol_shape, padding_factor)
    padded_half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(padded_shape)
    relion_ctf.ensure_device_headroom(int(_DEVICE_PAD_HALVES * np.prod(padded_half_shape) * 8))
    # A compact accumulator whose FFTW pad the device cannot hold (_relion_pad_exceeds_device_working_set) is solved
    # on the CPU backend from its repack on: after a pass the 16 GB pool could not place even the 0.69 GiB repacked
    # half at box 448 (relax#49). The accumulators go back to the device if the reconstruction is not host-staged.
    cpu_accumulators = not relion_functions._large_grid_postprocess_is_physically_large(
        int(np.prod(accumulator_volume_shape or [int(vol_shape[0]) * int(padding_factor)] * 3))
    ) and _relion_pad_exceeds_device_working_set(padded_shape)
    if cpu_accumulators:
        cpu = jax.devices("cpu")[0]
        Ft_ctf, Ft_y = (jax.device_put(value, cpu) if isinstance(value, jax.Array) else value for value in (Ft_ctf, Ft_y))
        with jax.default_device(cpu):
            Ft_ctf, Ft_y = _pack_compact_full_accumulators_for_large_relion_ifft(
                Ft_ctf, Ft_y, vol_shape, padding_factor, accumulator_volume_shape, relion_functions
            )
    else:
        Ft_ctf, Ft_y = _pack_compact_full_accumulators_for_large_relion_ifft(
            Ft_ctf,
            Ft_y,
            vol_shape,
            padding_factor,
            accumulator_volume_shape,
            relion_functions,
        )
    stable_class = (
        _stable_reconstruction_class(current_size, vol_shape, padding_factor, accumulator_volume_shape, tau_is_1d)
        if programs.stable_windows
        else None
    )
    unregularized_class = (
        _stable_unregularized_class(vol_shape, padding_factor, accumulator_volume_shape, tau, current_size)
        if programs.stable_windows
        else None
    )
    postprocess_args = (Ft_ctf, Ft_y, vol_shape, padding_factor)
    postprocess_kwargs = dict(
        tau=tau,
        kernel="triangular",
        use_spherical_mask=use_spherical_mask,
        grid_correct=grid_correct,
        gridding_correct=gridding_correct,
        kernel_width=1,
        tau2_fudge=tau2_fudge,
        gridding_padding_factor=projection_padding_factor,
        minres_map=minres_map,
        current_size=current_size,
        return_real_space=return_real_space,
        accumulator_volume_shape=accumulator_volume_shape,
        tau_is_1d=tau_is_1d,
        preserve_output_precision=preserve_output_precision,
        # EM transform precision follows its accumulation precision, not the
        # deliberate RFLOAT denominator/gridding operands. Complex128 inputs
        # retain the diagnostic transform path.
        fft_compute_dtype=jnp.result_type(Ft_y.dtype, jnp.complex64),
        relion_filter_scale=relion_filter_scale,
    )
    host_stage_large_ifft = _should_host_stage_large_relion_ifft(
        Ft_ctf,
        Ft_y,
        vol_shape,
        padding_factor,
        accumulator_volume_shape,
        relion_functions,
    )
    if cpu_accumulators and not host_stage_large_ifft:
        Ft_ctf, Ft_y = (jax.device_put(value, jax.devices()[0]) for value in (Ft_ctf, Ft_y))
    if retained_device_numerator is not None and not host_stage_large_ifft:
        raise ValueError(
            "A retained device numerator is only valid for the large host-staged RELION reconstruction path"
        )
    if not host_stage_large_ifft:
        if stable_class is not None:
            physical_size, physical_shape = stable_class
            postprocess_args = (
                *_pad_accumulators_to_class(
                    Ft_ctf,
                    Ft_y,
                    tuple(int(v) for v in accumulator_volume_shape),
                    tuple(int(v) for v in physical_shape),
                ),
                vol_shape,
                padding_factor,
            )
            postprocess_kwargs = dict(
                postprocess_kwargs,
                current_size=int(physical_size),
                accumulator_volume_shape=physical_shape,
                logical_current_size=jnp.int32(int(current_size)),
            )
        elif unregularized_class is not None:
            logical_shape = tuple(int(v) for v in accumulator_volume_shape)
            postprocess_args = (
                *_pad_accumulators_to_class(Ft_ctf, Ft_y, logical_shape, unregularized_class),
                vol_shape,
                padding_factor,
            )
            postprocess_kwargs = dict(
                postprocess_kwargs,
                accumulator_volume_shape=unregularized_class,
                logical_accumulator_size=jnp.int32(logical_shape[0]),
            )
        result = relion_functions.post_process_from_filter_v2(
            *postprocess_args,
            **postprocess_kwargs,
        )
        reconstruction_shape = relion_functions._relion_reconstruction_padded_shape(
            vol_shape,
            padding_factor,
        )
        if _large_irfft_requires_explicit_normalization(reconstruction_shape):
            # XLA's built-in ``norm='backward'`` normalization overflows its
            # signed-int32 transform-size product at 1600^3 and silently omits
            # the reciprocal.  Physically large device accumulators take this
            # monolithic branch to avoid overlapping another box-scale device
            # buffer, so apply the reciprocal to the completed result in a
            # separate donating executable.
            logger.info(
                "RELION large inverse-FFT normalization boundary: "
                "reconstruction_shape=%s transform_size=%d "
                "implementation=jax_monolithic_dynamic_scale",
                reconstruction_shape,
                math.prod(reconstruction_shape),
            )
            result = _apply_large_irfft_scale(result, reconstruction_shape)
        return result

    accumulator_shape = _accumulator_shape_or_default(vol_shape, padding_factor, accumulator_volume_shape)
    reconstruction_shape = relion_functions._relion_reconstruction_padded_shape(
        vol_shape,
        padding_factor,
    )
    packed_half_bytes = int(
        np.prod(fourier_transform_utils.volume_shape_to_half_volume_shape(reconstruction_shape))
        * np.dtype(np.complex64).itemsize
    )
    logger.info(
        "RELION split pre-IFFT host boundary: accumulator_shape=%s reconstruction_shape=%s packed_half_bytes=%d",
        accumulator_shape,
        reconstruction_shape,
        packed_half_bytes,
    )
    if accumulator_shape[0] > reconstruction_shape[0]:
        # The original Stage-A executable combined denominator
        # regularization and complex division.  Although donation aliases its
        # numerator to the output, the regularization needs several
        # box-scale float32 temporaries.  Keep the complex64 numerator on the
        # host while those temporaries are live, then stage it only for the
        # zero-temporary donating divide.
        stage_a_filter = jnp.asarray(Ft_ctf)
        stage_a_filter.block_until_ready()
        if np.dtype(stage_a_filter.dtype) != np.dtype(np.float32):
            raise TypeError(
                f"Large RELION Stage A requires a float32 filter for exact donation, got {stage_a_filter.dtype}"
            )
        logger.info(
            "RELION Stage A staging host filter for donating regularization: shape=%s dtype=%s",
            tuple(stage_a_filter.shape),
            stage_a_filter.dtype,
        )
        regularized_filter_device = relion_functions_relion._regularize_large_relion_half_filter_donate_ctf(
            stage_a_filter,
            tau,
            vol_shape,
            padding_factor,
            tau2_fudge,
            minres_map,
            current_size,
            accumulator_shape,
            tau_is_1d,
            relion_filter_scale,
        )
        regularized_filter_device.block_until_ready()
        filter_input_donated = _device_array_is_deleted(stage_a_filter)
        if filter_input_donated is not True:
            regularization_relion.delete_device_array(regularized_filter_device)
            regularization_relion.delete_device_array(stage_a_filter)
            raise RuntimeError("Large RELION Stage-A regularization did not donate its float32 filter input")
        if np.dtype(regularized_filter_device.dtype) != np.dtype(np.float32):
            regularization_relion.delete_device_array(regularized_filter_device)
            raise TypeError(
                f"Large RELION Stage-A regularization must return float32, got {regularized_filter_device.dtype}"
            )
        logger.info(
            "RELION Stage A regularization complete: filter_input_donated=%s",
            filter_input_donated,
        )

        if retained_device_numerator is not None:
            if tuple(retained_device_numerator.shape) != tuple(Ft_y.shape):
                raise ValueError(
                    "Retained device numerator shape does not match the host numerator: "
                    f"{tuple(retained_device_numerator.shape)} != {tuple(Ft_y.shape)}"
                )
            if np.dtype(retained_device_numerator.dtype) != np.dtype(Ft_y.dtype):
                raise ValueError(
                    "Retained device numerator dtype does not match the host numerator: "
                    f"{retained_device_numerator.dtype} != {Ft_y.dtype}"
                )
            stage_a_numerator = retained_device_numerator
            logger.info(
                "RELION Stage A reusing retained half-0 device numerator: shape=%s dtype=%s",
                tuple(retained_device_numerator.shape),
                retained_device_numerator.dtype,
            )
        else:
            # A NumPy argument passed directly to a donate_argnums JIT is first
            # staged by dispatch, but that transient input cannot be donated.
            # Materialise an explicit JAX array so half 2 can alias its 15-GiB
            # Stage-A output into the staged numerator just as half 1 aliases
            # the retained low-resolution-join buffer.
            stage_a_numerator = jnp.asarray(Ft_y)
            stage_a_numerator.block_until_ready()
            if isinstance(Ft_y, np.ndarray):
                logger.info(
                    "RELION Stage A staging host numerator for donation: shape=%s dtype=%s",
                    tuple(stage_a_numerator.shape),
                    stage_a_numerator.dtype,
                )
        wiener_half_device = relion_functions_relion._divide_large_relion_half_numerator_donate_numerator(
            stage_a_numerator,
            regularized_filter_device,
            padding_factor,
            current_size,
            accumulator_shape,
        )
        wiener_half_device.block_until_ready()
        wiener_half_host = np.asarray(jax.device_get(wiener_half_device)).reshape(
            fourier_transform_utils.volume_shape_to_half_volume_shape(accumulator_shape),
        )
        regularization_relion.delete_device_array(wiener_half_device)
        regularization_relion.delete_device_array(stage_a_numerator)
        regularization_relion.delete_device_array(regularized_filter_device)
        regularization_relion.delete_device_array(stage_a_filter)
        del wiener_half_device
        del stage_a_numerator
        del regularized_filter_device
        del stage_a_filter
        gc.collect()

        fftw_half_host = _crop_relion_wiener_half_to_fftw_host(
            wiener_half_host,
            accumulator_shape,
            reconstruction_shape,
            relion_functions,
        )
        del wiener_half_host
        gc.collect()
    else:
        if retained_device_numerator is not None:
            raise ValueError(
                "A retained device numerator is only valid for the crop branch of the "
                "large host-staged RELION reconstruction path"
            )
        if relion_functions._large_grid_postprocess_is_physically_large(
            int(np.prod(accumulator_shape)),
        ):
            # A physically large accumulator that is still smaller than the
            # inverse-FFT grid (EMPIAR-10202 from about current size 410: 823^3
            # to 1163^3 against 1600^3) returns its Wiener half before the pad,
            # and the pad into the 16.4-GB FFTW half runs on the host, as RELION
            # windows its reconstruction on the CPU. Padding on the device built
            # the zero grid and its scattered copy side by side (32.8 GB live in
            # the census, bigbox 14468686) on top of the resident state.
            wiener_half_device = relion_functions.post_process_from_filter_v2(
                *postprocess_args,
                **postprocess_kwargs,
                input_half_volume=True,
                return_wiener_half_before_window=True,
            )
            wiener_half_device.block_until_ready()
            wiener_half_host = np.asarray(jax.device_get(wiener_half_device))
            regularization_relion.delete_device_array(wiener_half_device)
            del wiener_half_device
            gc.collect()
            logger.info(
                "RELION pre-window Wiener half padded on the host: accumulator_shape=%s reconstruction_shape=%s",
                accumulator_shape,
                reconstruction_shape,
            )
            fftw_half_host = _pad_relion_wiener_half_to_fftw_host(
                wiener_half_host,
                accumulator_shape,
                reconstruction_shape,
                relion_functions,
            )
            del wiener_half_host
        elif _relion_pad_exceeds_device_working_set(reconstruction_shape):
            # The same Wiener solve and pad on the CPU backend: the device would need the zero FFTW half and its
            # scattered copy as single blocks (relax#49: at box 380 a 1.64 GiB FFTW half could not be placed in
            # a fragmented 16 GB pool; at box 448 2.69 GiB did not fit). Only the small accumulators move.
            cpu = jax.devices("cpu")[0]

            def on_cpu(value):
                return jax.device_put(value, cpu) if isinstance(value, jax.Array) else value

            with jax.default_device(cpu):
                fftw_half_host = np.asarray(
                    relion_functions.post_process_from_filter_v2(
                        *[on_cpu(value) for value in postprocess_args],
                        **{key: on_cpu(value) for key, value in postprocess_kwargs.items()},
                        input_half_volume=True,
                        return_fftw_half_before_ifft=True,
                    )
                )
            logger.info(
                "RELION Wiener solve and FFTW pad on the CPU backend: accumulator_shape=%s reconstruction_shape=%s",
                accumulator_shape,
                reconstruction_shape,
            )
        else:
            fftw_half_device = relion_functions.post_process_from_filter_v2(
                *postprocess_args,
                **postprocess_kwargs,
                input_half_volume=True,
                return_fftw_half_before_ifft=True,
            )
            fftw_half_device.block_until_ready()
            fftw_half_host = np.asarray(jax.device_get(fftw_half_device))
            del fftw_half_device
        gc.collect()

    explicit_irfft_normalization = _large_irfft_requires_explicit_normalization(
        reconstruction_shape,
    )
    host_irfft = _large_relion_host_irfft_enabled(reconstruction_shape, forced=programs.host_irfft)
    if host_irfft:
        logger.info(
            "RELION padded inverse FFT using host scipy.fft: reconstruction_shape=%s "
            "output_shape=%s input_bytes=%d workers=%d",
            reconstruction_shape,
            tuple(int(size) for size in vol_shape),
            int(fftw_half_host.nbytes),
            programs.host_fft_workers,
        )
        unpadded_real_host = _host_irfft_and_center_crop(
            fftw_half_host,
            reconstruction_shape,
            vol_shape,
            workers=programs.host_fft_workers,
        )
        del fftw_half_host
        gc.collect()
        result = relion_functions_relion._finish_large_relion_postprocess_from_unpadded_real(
            unpadded_real_host,
            vol_shape,
            padding_factor,
            kernel="triangular",
            use_spherical_mask=use_spherical_mask,
            grid_correct=grid_correct,
            gridding_correct=gridding_correct,
            kernel_width=1,
            return_real_space=return_real_space,
            gridding_padding_factor=projection_padding_factor,
        )
    else:
        # The device CTF row caches yield if the device cannot hand out the transform's working set now, as a
        # pass's accumulators make them yield (relax#40).
        relion_ctf.ensure_device_headroom(int(_DEVICE_IRFFT_HALVES * packed_half_bytes))
        result = relion_functions_relion._finish_large_relion_postprocess_from_fftw_half(
            fftw_half_host,
            vol_shape,
            padding_factor,
            kernel="triangular",
            use_spherical_mask=use_spherical_mask,
            grid_correct=grid_correct,
            gridding_correct=gridding_correct,
            kernel_width=1,
            return_real_space=return_real_space,
            gridding_padding_factor=projection_padding_factor,
        )
    if explicit_irfft_normalization:
        logger.info(
            "RELION large inverse-FFT normalization boundary: "
            "reconstruction_shape=%s transform_size=%d implementation=%s",
            reconstruction_shape,
            math.prod(reconstruction_shape),
            "scipy_host_backward" if host_irfft else "jax_dynamic_scale",
        )
    if explicit_irfft_normalization and not host_irfft:
        # The inverse-FFT executable has completed; see _apply_large_irfft_scale.
        result = _apply_large_irfft_scale(result, reconstruction_shape)
    return result


def _apply_relion_initial_lowpass_filter(
    volume_ft_flat, volume_shape, voxel_size, ini_high_angstrom, filter_edgewidth=5
):
    """Apply RELION's ``initialLowPassFilterReferences`` to a full Fourier volume."""
    if ini_high_angstrom is None or float(ini_high_angstrom) <= 0.0:
        return volume_ft_flat
    # RELION filters Iref in double on the host. Both transforms run there too, on the CPU backend, and only the
    # filtered map goes back to the device in its dtype: on the device the inverse transform's copies and the
    # complex128 forward transform held seven maps, 1.34 GiB requests at box 448, which a 16 GB V100 could not
    # place in iteration 1 (relax#47).
    dtype = volume_ft_flat.dtype
    with jax.default_device("cpu"):
        original = jnp.asarray(np.asarray(volume_ft_flat)).reshape(volume_shape)
        volume_real = np.real(np.asarray(fourier_transform_utils.get_idft3(original))).astype(np.float64, copy=False)
        del original
        filtered_real = initial_low_pass_filter_references(
            volume_real[None, ...],
            box_size=int(volume_shape[0]),
            pixel_size=float(voxel_size),
            ini_high_ang=float(ini_high_angstrom),
            filter_edgewidth=float(filter_edgewidth),
        )[0]
        del volume_real
        filtered_ft = np.asarray(fourier_transform_utils.get_dft3(jnp.asarray(filtered_real)).astype(dtype))
    return jnp.asarray(filtered_ft.reshape(-1))


@functools.partial(jax.jit, static_argnames=("volume_shape",))
def _centered_real_overlap(volume_ft_flat, reference_ft_flat, *, volume_shape):
    """The dot product of the two volumes' real parts in real space, each less its mean, on the device."""

    vol_real = jnp.real(fourier_transform_utils.get_idft3(volume_ft_flat.reshape(volume_shape)))
    ref_real = jnp.real(fourier_transform_utils.get_idft3(reference_ft_flat.reshape(volume_shape)))
    return jnp.sum((ref_real - jnp.mean(ref_real)) * (vol_real - jnp.mean(vol_real)))


def _align_fourier_volume_sign_to_reference(volume_ft_flat, reference_ft_flat, volume_shape):
    """Keep reconstructed volumes on the same real-space sign branch as the reference.

    Only the overlap's sign is read back: the volumes' inverse transforms, centring and dot product stay on the
    device (copying both real volumes to the host in float64 took 7.6 s of a 10k-particle box-256 auto-refine).
    """
    if reference_ft_flat is None:
        return volume_ft_flat, False
    shape = tuple(int(n) for n in volume_shape)
    overlap = float(_centered_real_overlap(jnp.asarray(volume_ft_flat), jnp.asarray(reference_ft_flat), volume_shape=shape))
    if overlap < 0.0:
        return -volume_ft_flat, True
    return volume_ft_flat, False


@dataclass(frozen=True, kw_only=True)
class ReconstructionSettings:
    """Run-level geometry, regularization, mask and initial-filter settings."""

    box_size: int
    voxel_size: float
    volume_shape: tuple
    padding_factor: int
    projection_padding_factor: int
    minres_map: int
    width_mask_edge: int
    fmask_edge: int
    tau2_fudge: float
    particle_diameter_angstrom: float | None
    first_iteration_lowpass_angstrom: float | None
    # Real-space gridding-correction window of every reconstruction ("radial" is RELION's);
    # "separable" is K=1 only and the class operations refuse it.
    gridding_kernel: str = "radial"
    # How the 3-D shell statistics behind tau2, data-vs-prior and the half-map FSC count Hermitian
    # pairs: "relion" counts those of the stored half's zero plane twice, "once" every pair once.
    # The 1/1000 weight floor inside the reconstruction (RECOVAR) keeps RELION's counting.
    shell_pair_counting: str = "relion"
    # RELION --solvent_mask: the user reference mask on the model grid in the internal (z, y, x)
    # frame (relax.reconstruction.solvent_mask.read_solvent_mask, transposed); it replaces the
    # particle-diameter sphere of the solvent flatten. None keeps the sphere.
    solvent_mask: object = field(default=None, compare=False, repr=False)
    # RELION --solvent_correct_fsc (MPI relion_refine): the K=1 half-set FSC is the masked,
    # phase-randomisation corrected FSC of the unregularised half maps; needs solvent_mask.
    solvent_correct_fsc: bool = False
    # Seed of the corrected FSC's random phases, drawn per iteration.
    solvent_fsc_seed: int = 0
    # The run's reconstruction programs (ScoringVariants.reconstruction): no effect on values.
    programs: ReconstructionPrograms

    def __post_init__(self):
        # Python floats, so the solvent-mask radius is the same double arithmetic for every caller.
        object.__setattr__(self, "voxel_size", float(self.voxel_size))
        if self.particle_diameter_angstrom is not None:
            object.__setattr__(self, "particle_diameter_angstrom", float(self.particle_diameter_angstrom))
        if self.solvent_correct_fsc and self.solvent_mask is None:
            raise ValueError("--solvent_correct_fsc needs --solvent_mask (RELION corrects only with a user mask)")
        if self.solvent_mask is not None and tuple(np.shape(self.solvent_mask)) != tuple(self.volume_shape):
            raise ValueError(
                f"solvent mask shape {np.shape(self.solvent_mask)} is not the model's {tuple(self.volume_shape)}"
            )

    def reconstruct(self, Ft_ctf, Ft_y, *, tau, current_size, accumulator_volume_shape, **solve_options):
        """One map from one accumulator pair with this run's reconstruction settings.

        Forwards ``volume_shape``, ``padding_factor``, ``tau2_fudge``, ``projection_padding_factor``,
        ``minres_map`` and ``gridding_kernel`` to ``_reconstruct_volume_eager``; ``solve_options`` are its
        per-call options (``tau_is_1d``, ``preserve_output_precision``, ...).
        """
        return _reconstruct_volume_eager(
            Ft_ctf,
            Ft_y,
            self.volume_shape,
            self.padding_factor,
            tau=tau,
            tau2_fudge=self.tau2_fudge,
            projection_padding_factor=self.projection_padding_factor,
            minres_map=self.minres_map,
            current_size=current_size,
            accumulator_volume_shape=accumulator_volume_shape,
            gridding_kernel=self.gridding_kernel,
            programs=self.programs,
            **solve_options,
        )


def _require_radial_gridding_for_classes(settings: ReconstructionSettings) -> None:
    """Class3D keeps RELION's radial window: its tau2 is the power of the radially corrected reference."""

    if settings.gridding_kernel != "radial":
        raise NotImplementedError(
            f"gridding_kernel={settings.gridding_kernel!r} is K=1 only; class reconstructions keep the radial window"
        )


@dataclass(frozen=True)
class SplitHalfPrior:
    """K1 scoring/reconstruction priors and the shared FSC, which also drives tau2 and size growth."""

    variance: object
    variance_per_half: list
    shells_per_half: list
    fsc: object
    details_per_half: list[dict]


def estimate_split_half_prior(
    numerators,
    denominators,
    settings: ReconstructionSettings,
    *,
    current_size,
    accumulator_shape,
    full_half_axes,
    iteration,
    scoring_dtype,
    log,
) -> SplitHalfPrior:
    """Estimate independent half priors from this expectation's shared FSC.

    The raw backprojector FSC is reported and drives tau2 and size growth.
    The two halves retain their own
    Fourier weights.
    See ``docs/math/relion_refinement_algorithm.md`` for the M-step ordering.
    """

    fsc_clock = Stopwatch()
    current_iter_fsc = regularization_relion.compute_relion_fsc_from_backprojector(
        numerators[0],
        numerators[1],
        denominators[0],
        denominators[1],
        settings.volume_shape,
        padding_factor=settings.padding_factor,
        r_max=current_size // 2,
        accumulator_volume_shape=accumulator_shape,
        output_dtype=scoring_dtype,
        shell_pair_counting=settings.shell_pair_counting,
    )
    log.info(
        "Computed iter-%d FSC for tau2 (RELION backprojector path): %.1fs",
        iteration + 1,
        fsc_clock.seconds,
    )
    if settings.solvent_correct_fsc:
        current_iter_fsc = _solvent_corrected_fsc(
            numerators,
            denominators,
            settings,
            current_size=current_size,
            accumulator_shape=accumulator_shape,
            iteration=iteration,
            like=current_iter_fsc,
            log=log,
        )

    # RELION calls BackProjector::updateSSNRarrays independently for each
    # half-map BPref.  The gold-standard FSC is shared, but sigma2/tau2
    # come from each half's own Fourier weight outside the joined shells.
    tau2_update_details_per_half = []
    mean_signal_variance_per_half = []
    for half_idx, Ft_ctf_half in enumerate((denominators[0], denominators[1])):
        full_half_axis = full_half_axes[half_idx]
        mean_signal_variance_k, _, tau2_update_details_k = regularization_relion.compute_relion_tau2_from_weights(
            Ft_ctf_half,
            Ft_ctf_half,
            current_iter_fsc,
            settings.volume_shape,
            tau2_fudge=settings.tau2_fudge,
            padding_factor=settings.padding_factor,
            r_max=current_size // 2,
            return_details=True,
            full_half_axis=-1 if full_half_axis is None else int(full_half_axis),
            accumulator_volume_shape=accumulator_shape,
            output_dtype=scoring_dtype,
            shell_pair_counting=settings.shell_pair_counting,
        )
        mean_signal_variance_per_half.append(mean_signal_variance_k)
        tau2_update_details_per_half.append(tau2_update_details_k)
    mean_signal_variance_shells_per_half = [details["prior_shells"] for details in tau2_update_details_per_half]
    mean_signal_variance = 0.5 * (mean_signal_variance_per_half[0] + mean_signal_variance_per_half[1])
    return SplitHalfPrior(
        variance=mean_signal_variance,
        variance_per_half=mean_signal_variance_per_half,
        shells_per_half=mean_signal_variance_shells_per_half,
        fsc=current_iter_fsc,
        details_per_half=tau2_update_details_per_half,
    )


def _solvent_corrected_fsc(
    numerators, denominators, settings, *, current_size, accumulator_shape, iteration, like, log
):
    """RELION's --solvent_correct_fsc curve in place of the backprojector FSC ``like``.

    The unregularised half maps are those relax writes as run_itNNN_half*_unfil.mrc
    (``reconstruct_unregularized_k1_halfmaps``), made from this iteration's accumulators after the
    low-resolution join, as RELION reconstructs its BPref copies (ml_optimiser_mpi.cpp:3221-3300).
    """

    from relax.reconstruction.solvent_mask import solvent_corrected_fsc

    clock = Stopwatch()
    unregularised = reconstruct_unregularized_k1_halfmaps(
        numerators,
        denominators,
        settings,
        accumulator_volume_shape=accumulator_shape,
    )
    # RELION's getFSC sums the FFTW half spectrum halved along the map file's x axis, and the
    # half-spectrum sum depends on which axis is halved; the maps and the mask return to the file's
    # axis order (the transpose of the internal frame; relax.helpers.map_io).
    half1, half2 = (
        np.transpose(
            np.real(np.asarray(fourier_transform_utils.get_idft3(jnp.asarray(m).reshape(settings.volume_shape)))),
            (2, 1, 0),
        )
        for m in unregularised
    )
    fsc, details = solvent_corrected_fsc(
        half1,
        half2,
        np.transpose(np.asarray(settings.solvent_mask, dtype=np.float64), (2, 1, 0)),
        current_size=current_size,
        rng=np.random.default_rng((int(settings.solvent_fsc_seed), int(iteration))),
    )
    like = np.asarray(like)
    if fsc.shape != like.shape:
        raise ValueError(f"the corrected FSC has {fsc.shape[0]} shells, the backprojector FSC {like.shape[0]}")
    log.info(
        "iter-%d solvent-corrected FSC: randomize phases beyond shell %d (%.2f A): %.1fs",
        iteration + 1,
        details["randomize_at"],
        settings.box_size * settings.voxel_size / max(details["randomize_at"], 1),
        clock.seconds,
    )
    return jnp.asarray(fsc, dtype=like.dtype)


def _reconstruct_k1_maps(
    numerators_by_half,
    denominators_by_half,
    tau_by_half,
    settings: ReconstructionSettings,
    *,
    current_size,
    accumulator_volume_shape,
    retained_first_numerator=None,
) -> list:
    """Reconstruct both K=1 halves while preserving RELION buffer lifetime."""

    cs_int = int(current_size) if current_size is not None else None
    reconstructed_means = []
    retained_device_numerator = retained_first_numerator
    for k, (Ft_y_half, Ft_ctf_half, tau_half) in enumerate(zip(numerators_by_half, denominators_by_half, tau_by_half)):
        # This RELION build uses double RFLOAT in BackProjector::reconstruct.
        # Keep the stored/controller tau2 state compact, but promote the
        # reconstruction operand so 1 / (padding_factor**3 * tau2) is not
        # rounded in float32 before it enters the Wiener denominator.
        reconstruction_tau = jnp.asarray(tau_half, dtype=jnp.float64)
        reconstructed = settings.reconstruct(
            Ft_ctf_half,
            Ft_y_half,
            tau=reconstruction_tau,
            current_size=cs_int,
            accumulator_volume_shape=accumulator_volume_shape,
            tau_is_1d=True,
            preserve_output_precision=True,
            relion_filter_scale=float(settings.volume_shape[0] ** 4),
            **(
                {"retained_device_numerator": retained_device_numerator}
                if k == 0 and retained_device_numerator is not None
                else {}
            ),
        ).reshape(-1)
        reconstructed_means.append(_finish_host_staged_reconstruction(reconstructed, Ft_ctf_half, Ft_y_half))
        if k == 0 and retained_device_numerator is not None:
            retained_device_numerator = None
            gc.collect()
    return reconstructed_means


def _reconstruct_class_maps(
    combined_numerators,
    combined_denominators,
    tau_by_class,
    settings: ReconstructionSettings,
    *,
    n_classes,
    iteration,
    current_size,
    accumulator_volume_shape,
):
    """Reconstruct the shared Class3D stack from combined accumulators."""

    _require_radial_gridding_for_classes(settings)
    clock = Stopwatch()
    cs_int = int(current_size) if current_size is not None else None
    shared_class_maps = []
    for class_idx in range(n_classes):
        logger.info(
            "Class3D reconstruction start: iter=%d class=%d/%d current_size=%s",
            iteration + 1,
            class_idx + 1,
            n_classes,
            cs_int,
        )
        class_map = settings.reconstruct(
            combined_denominators[class_idx],
            combined_numerators[class_idx],
            tau=tau_by_class[class_idx],
            current_size=cs_int,
            accumulator_volume_shape=accumulator_volume_shape,
            tau_is_1d=True,
        ).reshape(-1)
        shared_class_maps.append(class_map)
        logger.info(
            "Class3D reconstruction done: iter=%d class=%d/%d elapsed=%.1fs",
            iteration + 1,
            class_idx + 1,
            n_classes,
            clock.seconds,
        )
    shared_classes = jnp.stack(shared_class_maps, axis=0)
    logger.info(
        "Class3D reconstruction stack complete: iter=%d classes=%d elapsed=%.1fs",
        iteration + 1,
        n_classes,
        clock.seconds,
    )
    return shared_classes


def _solvent_flatten_requested(settings: ReconstructionSettings) -> bool:
    """Return whether new references are solvent-flattened: a user mask or a particle diameter is set."""
    if settings.solvent_mask is not None:
        return True
    return settings.particle_diameter_angstrom is not None and settings.particle_diameter_angstrom > 0


def _numbered_solvent_mask(settings: ReconstructionSettings, *, dtype):
    """The solvent mask in ``dtype``: the user mask, else the soft sphere of the particle diameter.

    RELION's solventFlatten multiplies each reference by the user mask as read (no soft edge
    added), or by the cosine-edged sphere when there is none (ml_optimiser.cpp:5506-5590).
    """
    if settings.solvent_mask is not None:
        return jnp.asarray(settings.solvent_mask, dtype=dtype)
    flatten_radius = settings.particle_diameter_angstrom / (2.0 * settings.voxel_size)
    return _make_relion_solvent_mask(
        settings.volume_shape,
        radius=flatten_radius,
        radius_p=flatten_radius + settings.width_mask_edge,
        dtype=dtype,
    )


@functools.partial(jax.jit, donate_argnums=0)
def _set_class_row(class_maps, row, class_idx):
    """``class_maps`` with row ``class_idx`` replaced by ``row``, in place (the stack is donated)."""
    return class_maps.at[class_idx].set(row)


def _lowpass_class_stack(class_maps, settings: ReconstructionSettings, n_classes):
    """Apply the first-CC initial low-pass to every class, in place (``class_maps`` is consumed).

    One class is filtered at a time and written back into the stack, so the stack and one class's temporaries
    are live. Building the classes in a list and stacking them held the input, the classes and the new stack
    together: 1.23 GiB each for Class3D K=3 at box 380 (relax#45).
    """
    for class_idx in range(n_classes):
        row = _apply_relion_initial_lowpass_filter(
            class_maps[class_idx],
            settings.volume_shape,
            settings.voxel_size,
            settings.first_iteration_lowpass_angstrom,
            filter_edgewidth=settings.fmask_edge,
        )
        class_maps = _set_class_row(class_maps, row, class_idx)
        del row
    return class_maps


@functools.partial(jax.jit, static_argnames=("volume_shape",))
def _flatten_volume(volume_ft_flat, solvent_mask, *, volume_shape):
    """RELION's solventFlatten of one flat Fourier map as one program: inverse transform, mask, forward transform."""
    vol_real = fourier_transform_utils.get_idft3(volume_ft_flat.reshape(volume_shape))
    return fourier_transform_utils.get_dft3(vol_real * solvent_mask).reshape(-1)


@functools.partial(jax.jit, static_argnames=("volume_shape",), donate_argnums=0)
def _flatten_class_row(class_maps, solvent_mask, class_idx, *, volume_shape):
    """``class_maps`` with class ``class_idx`` solvent-flattened, in place (the stack is donated)."""
    return class_maps.at[class_idx].set(_flatten_volume(class_maps[class_idx], solvent_mask, volume_shape=volume_shape))


def _flatten_class_stack(class_maps, solvent_mask, volume_shape, n_classes):
    """Solvent-flatten every class on the device with one mask, in place (``class_maps`` is consumed).

    One program per class keeps the stack plus about two class maps of transforms live (measured at box 256:
    2.0 maps; the eager statements held 5.0, and collecting the classes in a list then stacking them 11.8).
    """
    volume_shape = tuple(int(n) for n in volume_shape)
    for class_idx in range(n_classes):
        class_maps = _flatten_class_row(class_maps, solvent_mask, class_idx, volume_shape=volume_shape)
    return class_maps


def _log_first_cc_lowpass(settings: ReconstructionSettings) -> None:
    """Report the first-CC initial low-pass once all slots are postprocessed."""
    if settings.first_iteration_lowpass_angstrom is not None:
        logger.info(
            "RELION iter-1 CC emulation: reapplying ini_high low-pass filter at %.2f A",
            float(settings.first_iteration_lowpass_angstrom),
        )


def reconstruct_numbered_k1_halfmaps(
    numerators_by_half,
    denominators_by_half,
    tau_by_half,
    settings: ReconstructionSettings,
    *,
    iteration,
    current_size,
    accumulator_volume_shape,
    relion_firstiter_cc_this_iter,
    retained_first_numerator=None,
    observer: RunObserver | None = None,
) -> list:
    """Solve two independent numbered K1 maps, then postprocess each half.

    Numerators, denominators and priors are ordered half pairs; each prior is
    a shell curve. The private solve frame owns promoted priors, host
    completion and the retained half-0 numerator boundary. Both solves finish
    before each half is captured, first-CC filtered and solvent-flattened in
    turn. The flatten host-stages box-scale results and consumes its mask.
    Return ready maps for installation.
    """
    observer = RunObserver() if observer is None else observer
    means = _reconstruct_k1_maps(
        numerators_by_half,
        denominators_by_half,
        tau_by_half,
        settings,
        current_size=current_size,
        accumulator_volume_shape=accumulator_volume_shape,
        retained_first_numerator=retained_first_numerator,
    )
    for k in range(2):
        observer.map_solved(iteration, k, means[k], settings=settings, current_size=current_size, n_classes=1)
        # RELION filters Iref inside maximizationOtherParameters, then calls
        # solventFlatten from the outer iteration loop.  These operations do
        # not commute: masking in real space after the Fourier low-pass adds a
        # small, deterministic high-shell tail.
        if relion_firstiter_cc_this_iter:
            means[k] = _apply_relion_initial_lowpass_filter(
                means[k],
                settings.volume_shape,
                settings.voxel_size,
                settings.first_iteration_lowpass_angstrom,
                filter_edgewidth=settings.fmask_edge,
            )
        if _solvent_flatten_requested(settings):
            solvent_mask = _numbered_solvent_mask(settings, dtype=means[k].real.dtype)
            means[k] = _apply_relion_solvent_flatten_k1(
                means[k],
                solvent_mask,
                settings.volume_shape,
            )
    if relion_firstiter_cc_this_iter:
        _log_first_cc_lowpass(settings)
    return means


def reconstruct_numbered_class_maps(
    combined_numerators,
    combined_denominators,
    tau_by_class,
    settings: ReconstructionSettings,
    *,
    n_classes,
    iteration,
    current_size,
    accumulator_volume_shape,
    relion_firstiter_cc_this_iter,
    observer: RunObserver | None = None,
) -> list:
    """Solve one numbered Class3D reference stack from combined partitions.

    Accumulators and priors have a leading class axis; each prior is a shell
    curve. All class solves finish before premask capture, initial filtering
    and solvent flattening.
    Return a two-entry list of particle-execution slots, not scientific
    halves; both entries alias one stack. Each slot is captured, then every
    class is first-CC filtered, then every class is flattened on the device
    with one mask. Both slots hold the same stack and the steps are the same,
    so they run once, in place.
    """
    observer = RunObserver() if observer is None else observer
    shared_classes = _reconstruct_class_maps(
        combined_numerators,
        combined_denominators,
        tau_by_class,
        settings,
        n_classes=n_classes,
        iteration=iteration,
        current_size=current_size,
        accumulator_volume_shape=accumulator_volume_shape,
    )
    for k in range(2):
        observer.map_solved(
            iteration, k, shared_classes, settings=settings, current_size=current_size, n_classes=n_classes
        )
    # As for K1, the low-pass precedes the solvent flatten and does not commute with it.
    if relion_firstiter_cc_this_iter:
        shared_classes = _lowpass_class_stack(shared_classes, settings, n_classes)
    if _solvent_flatten_requested(settings):
        solvent_mask = _numbered_solvent_mask(settings, dtype=jnp.finfo(shared_classes.dtype).dtype)
        shared_classes = _flatten_class_stack(shared_classes, solvent_mask, settings.volume_shape, n_classes)
    if relion_firstiter_cc_this_iter:
        _log_first_cc_lowpass(settings)
    return [shared_classes, shared_classes]


@dataclass(frozen=True)
class K1ReportingPrior:
    """Tapered K1 shared/half volumes and their per-half reporting details."""

    variance: object
    variance_per_half: list
    details_per_half: list[dict]


def taper_first_cc_k1_prior(
    variance_per_half,
    details_per_half,
    settings: ReconstructionSettings,
    *,
    pixel_size_angstrom,
    scoring_dtype,
) -> K1ReportingPrior:
    """Taper K1 reporting priors after the untapered regularized reconstruction.

    Updates ``variance_per_half`` and ``details_per_half`` in place, preserving their identity and the
    half -> prior/SSNR update order. The expanded taper and radial grid are temporary implementation
    arrays; model installation and shared/per-half policy stay with the caller.
    """
    tau2_taper = _firstiter_cc_ini_high_tau2_taper(
        len(details_per_half[0]["prior_shells"]),
        settings.box_size,
        pixel_size_angstrom,
        settings.first_iteration_lowpass_angstrom,
        filter_edgewidth=settings.fmask_edge,
    )
    radial_shells = np.asarray(
        fourier_transform_utils.get_grid_of_radial_distances(
            settings.volume_shape,
            scaled=False,
            frequency_shift=0,
        ),
        dtype=np.int32,
    ).reshape(-1)
    radial_shells = np.minimum(radial_shells, len(tau2_taper) - 1)
    tau2_taper_volume = jnp.asarray(
        tau2_taper[radial_shells],
        dtype=scoring_dtype,
    )
    for half_idx in range(2):
        variance_per_half[half_idx] = variance_per_half[half_idx] * tau2_taper_volume
        for key in ("prior_shells", "ssnr_shells"):
            field_values = details_per_half[half_idx][key]
            details_per_half[half_idx][key] = field_values * jnp.asarray(
                tau2_taper,
                dtype=field_values.dtype,
            )
    variance = 0.5 * (variance_per_half[0] + variance_per_half[1])
    return K1ReportingPrior(variance, variance_per_half, details_per_half)


@dataclass(frozen=True)
class ClassReportingPrior:
    """Tapered Class3D shell stack and its aggregate prior/SSNR details."""

    shells: object
    details: dict


def taper_first_cc_class_prior(
    shells,
    details,
    settings: ReconstructionSettings,
    *,
    pixel_size_angstrom,
) -> ClassReportingPrior:
    """Taper class reporting shells after the caller publishes the tapered curve.

    The controller first tapers/publishes data-vs-prior for scheduling. This
    operation then adapts the shell stack and aggregate detail arrays in their
    existing order, in place, preserving the detail mapping's identity.
    """

    def taper_shells(values):
        return _firstiter_cc_ini_high_tapered(
            values,
            settings.box_size,
            pixel_size_angstrom,
            settings.first_iteration_lowpass_angstrom,
            filter_edgewidth=settings.fmask_edge,
        )

    shells = jnp.asarray(taper_shells(np.asarray(shells)))
    for key in ("prior_shells", "ssnr_shells"):
        details[key] = taper_shells(np.asarray(details[key]))
    return ClassReportingPrior(shells, details)


# ---------------------------------------------------------------------------
# Unregularized half-map reconstruction + sign alignment
# ---------------------------------------------------------------------------


def reconstruct_unregularized_k1_halfmaps(
    Ft_y_per_half,
    Ft_ctf_per_half,
    settings: ReconstructionSettings,
    *,
    accumulator_volume_shape=None,
) -> list:
    """Reconstruct each K=1 half from its own unregularized accumulator."""

    return [
        settings.reconstruct(
            Ft_ctf_half, Ft_y_half, tau=None, current_size=None, accumulator_volume_shape=accumulator_volume_shape
        )
        for Ft_ctf_half, Ft_y_half in zip(Ft_ctf_per_half, Ft_y_per_half)
    ]


def reconstruct_unregularized_class_means(
    Ft_y_combined,
    Ft_ctf_combined,
    settings: ReconstructionSettings,
    n_classes,
    *,
    accumulator_volume_shape=None,
) -> list:
    """Reconstruct the shared K-class stack from combined accumulators."""

    _require_radial_gridding_for_classes(settings)
    unreg_shared = jnp.stack(
        [
            settings.reconstruct(
                Ft_ctf_combined[class_idx],
                Ft_y_combined[class_idx],
                tau=None,
                current_size=None,
                accumulator_volume_shape=accumulator_volume_shape,
            ).reshape(-1)
            for class_idx in range(n_classes)
        ],
        axis=0,
    )
    return [unreg_shared, unreg_shared]


def align_k1_volume_signs(means, previous_means, unregularized_means, volume_shape) -> None:
    """Align K=1 means and matching diagnostic maps to previous references, in place (``means[k]`` and
    ``unregularized_means[k]`` are replaced)."""

    for k in range(2):
        means[k], sign_flipped = _align_fourier_volume_sign_to_reference(
            means[k],
            previous_means[k],
            volume_shape,
        )
        if sign_flipped and unregularized_means[k] is not None:
            unregularized_means[k] = -unregularized_means[k]
        if sign_flipped:
            logger.info("Aligned half-%d volume sign to the previous reference", k + 1)


def _large_irfft_requires_explicit_normalization(volume_shape) -> bool:
    """Return whether XLA's inverse-FFT normalization exceeds int32 range."""

    return math.prod(int(size) for size in volume_shape) > _LARGE_IRFFT_TRANSFORM_SIZE_LIMIT


# The solvent flatten program of one complex64 map holds about two maps on the device (measured 2.0 class maps at
# box 256; the eager statements held 5.0; test_class_stack_postprocess.py).
_DEVICE_FLATTEN_MAPS = 2.0

# Padding the Wiener half into the FFTW half on the device holds the zero FFTW half and its scattered copy side by
# side (bigbox 14468686): two halves.
_DEVICE_PAD_HALVES = 2.0

# The device inverse FFT's working set over its packed half: the half itself, the real output (twice the half's
# float32 count, about one more half in bytes) and the cuFFT work area, measured at 1.63 GiB for the 760^3 half of
# 1.76 GB (0.93 of it; relax#40), so three halves.
_DEVICE_IRFFT_HALVES = 3.0


def _device_allocator_limit_bytes() -> int | None:
    """The JAX allocator's byte limit on the first GPU, or None off GPU."""

    devices = [device for device in jax.devices() if getattr(device, "platform", "") in {"gpu", "cuda"}]
    if not devices:
        return None
    stats = devices[0].memory_stats() or {}
    limit = stats.get("bytes_limit")
    return None if limit is None else int(limit)


def _relion_pad_exceeds_device_working_set(reconstruction_shape, *, allocator_limit_bytes: int | None = None) -> bool:
    """Whether the Wiener solve and pad into the FFTW half of ``reconstruction_shape`` should run on the CPU.

    When the pad's device working set (``_DEVICE_PAD_HALVES`` packed complex64 halves) exceeds the single working
    set the allocator's limit allows (``xla_memory_reserve.single_working_set_bytes``; the limit is
    ``allocator_limit_bytes``, read from the device when None). Off GPU it is False.
    """

    limit = _device_allocator_limit_bytes() if allocator_limit_bytes is None else int(allocator_limit_bytes)
    if limit is None:
        return False
    half_bytes = int(np.prod(fourier_transform_utils.volume_shape_to_half_volume_shape(reconstruction_shape))) * 8
    return _DEVICE_PAD_HALVES * half_bytes > single_working_set_bytes(limit)


def _large_relion_host_irfft_enabled(
    volume_shape, *, forced: bool | None, allocator_limit_bytes: int | None = None
) -> bool:
    """Return whether a padded RELION inverse FFT should execute on the host.

    ``forced`` (``ReconstructionPrograms.host_irfft``) decides when not None. Otherwise automatically when the
    transform's int32 size product overflows, or when its device working set (``_DEVICE_IRFFT_HALVES`` packed
    complex64 halves) exceeds the single working set the allocator's limit allows
    (``xla_memory_reserve.single_working_set_bytes``; the limit is ``allocator_limit_bytes``, read from the device
    when None): a cuFFT work area that cannot be found aborts the process instead of raising.
    """

    if forced is not None:
        return forced
    if _large_irfft_requires_explicit_normalization(volume_shape):
        return True
    limit = _device_allocator_limit_bytes() if allocator_limit_bytes is None else int(allocator_limit_bytes)
    if limit is None:
        return False
    half_bytes = int(np.prod(fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape))) * 8
    working_set = _DEVICE_IRFFT_HALVES * half_bytes
    if working_set <= single_working_set_bytes(limit):
        return False
    logger.info(
        "RELION padded inverse FFT on the host: its device working set %.2f GiB exceeds %.0f%% of the %.2f GiB "
        "allocator limit (relax#40)",
        working_set / 2**30, 100 * SINGLE_WORKING_SET_LIMIT_SHARE, limit / 2**30,
    )
    return True


def _host_irfft_and_center_crop(
    fftw_half,
    reconstruction_shape,
    output_shape,
    *,
    workers: int,
):
    """Run a normalized c64-to-f32 inverse FFT and retain only its center crop.

    ``fftw_half`` is already in raw FFTW order.  The crop indices combine
    ``ifftshift`` with RELION's spatial unpadding so the host never allocates a
    second reconstruction-sized real volume merely to shift it.
    """

    from scipy import fft as scipy_fft

    reconstruction_shape = tuple(int(size) for size in reconstruction_shape)
    output_shape = tuple(int(size) for size in output_shape)
    if len(reconstruction_shape) != 3 or len(output_shape) != 3:
        raise ValueError(
            "RELION host inverse FFT requires three-dimensional shapes, got "
            f"reconstruction={reconstruction_shape} output={output_shape}"
        )
    if any(output > reconstruction for output, reconstruction in zip(output_shape, reconstruction_shape)):
        raise ValueError(
            "RELION host inverse FFT crop cannot exceed its reconstruction: "
            f"reconstruction={reconstruction_shape} output={output_shape}"
        )

    expected_half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(
        reconstruction_shape,
    )
    fftw_half = np.asarray(fftw_half, dtype=np.complex64, order="C").reshape(
        expected_half_shape,
    )
    real_raw = scipy_fft.irfftn(
        fftw_half,
        s=reconstruction_shape,
        axes=(-3, -2, -1),
        norm="backward",
        overwrite_x=True,
        workers=workers,
    )
    if real_raw.dtype != np.float32:
        raise TypeError(f"RELION host inverse FFT returned {real_raw.dtype}, expected float32")

    raw_indices = []
    for reconstruction, output in zip(reconstruction_shape, output_shape):
        padding_width = reconstruction - output
        pad_before = padding_width // 2
        centered_indices = np.arange(pad_before, pad_before + output, dtype=np.intp)
        # ``np.fft.ifftshift`` takes centered output index ``i`` from raw
        # input index ``i + floor(N / 2)``.  Keep the explicit floor because
        # the distinction matters for odd reconstruction sizes.
        raw_indices.append((centered_indices + reconstruction // 2) % reconstruction)
    cropped = np.asarray(
        real_raw[np.ix_(*raw_indices)],
        dtype=np.float32,
        order="C",
    )
    return cropped


def _accumulator_shape_or_default(vol_shape, padding_factor, accumulator_volume_shape) -> tuple:
    """The backprojector accumulator's full shape: ``accumulator_volume_shape``, else the padded box cubed."""
    return (
        tuple(3 * [int(vol_shape[0]) * int(padding_factor)])
        if accumulator_volume_shape is None
        else tuple(int(s) for s in accumulator_volume_shape)
    )


def _apply_large_irfft_scale(result, reconstruction_shape):
    """Apply the inverse FFT's 1 / N in a separate donating executable.

    XLA's built-in ``norm='backward'`` normalization overflows its signed-int32 transform-size product at
    1600^3 and silently omits the reciprocal; donation keeps this correction memory-neutral for box-scale maps.
    """
    inverse_transform_scale = jnp.asarray(
        np.float32(1.0 / float(math.prod(reconstruction_shape))),
    )
    return _normalize_large_irfft_result_donate(
        result,
        inverse_transform_scale,
    )


@functools.partial(jax.jit, donate_argnums=(0,))
def _normalize_large_irfft_result_donate(result, inverse_transform_scale):
    """Normalize a large raw inverse FFT in a separate donating executable."""

    return result * inverse_transform_scale


def _should_host_stage_large_relion_ifft(
    Ft_ctf,
    Ft_y,
    vol_shape,
    padding_factor,
    accumulator_volume_shape,
    relion_functions,
):
    """Return whether eager reconstruction should cross the padded-iFFT host boundary."""

    accumulator_shape = _accumulator_shape_or_default(vol_shape, padding_factor, accumulator_volume_shape)
    reconstruction_shape = relion_functions._relion_reconstruction_padded_shape(
        vol_shape,
        padding_factor,
    )
    if accumulator_shape == reconstruction_shape:
        return False
    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(accumulator_shape)
    half_size = int(np.prod(half_shape))

    def _is_packed_half(array):
        return tuple(array.shape) == half_shape or (array.ndim == 1 and int(array.size) == half_size)

    # The split is governed by the inverse-FFT grid, not the current-size
    # accumulator. Early box-scale iterations can have a compact accumulator
    # but still pad to a 1600^3 transform whose built-in normalization
    # overflows XLA's signed-int32 transform-size product. Compact device
    # accumulators can safely form and host-stage that one padded boundary.
    # Large accumulators still require the existing earlier host offload so
    # their storage cannot overlap the padded inverse-FFT workspace.
    accumulator_is_large = relion_functions._large_grid_postprocess_is_physically_large(
        int(np.prod(accumulator_shape)),
    )
    inputs_are_host = isinstance(Ft_ctf, np.ndarray) and isinstance(Ft_y, np.ndarray)
    if not _is_packed_half(Ft_ctf) or not _is_packed_half(Ft_y):
        return False
    if accumulator_is_large and not inputs_are_host:
        return False
    return relion_functions._large_grid_postprocess_single_precision_enabled(
        int(np.prod(reconstruction_shape)),
    )


def _pack_compact_full_accumulators_for_large_relion_ifft(
    Ft_ctf,
    Ft_y,
    vol_shape,
    padding_factor,
    accumulator_volume_shape,
    relion_functions,
):
    """Losslessly repack compact full accumulators before a giant padded iFFT.

    The RELION x-half M-step keeps the historical RECOVAR full-volume public
    contract on normal-sized accumulator grids. At large box sizes an early
    iteration can therefore reach reconstruction with compact full Hermitian
    arrays even though its padded inverse-FFT grid is giant. Repack only this
    compact/full case so it can use the packed pre-iFFT host boundary without
    changing the public M-step contract or the large-accumulator offload path.
    """

    accumulator_shape = _accumulator_shape_or_default(vol_shape, padding_factor, accumulator_volume_shape)
    reconstruction_shape = relion_functions._relion_reconstruction_padded_shape(
        vol_shape,
        padding_factor,
    )
    if accumulator_shape == reconstruction_shape:
        return Ft_ctf, Ft_y

    accumulator_voxels = int(np.prod(accumulator_shape))
    reconstruction_voxels = int(np.prod(reconstruction_shape))
    # The accumulator decision is physical: forcing single precision on a
    # compact grid must not misclassify it as too large to repack.  The
    # reconstruction still needs the single-precision large-grid path because
    # this boundary stages a complex64 packed half-volume.
    if relion_functions._large_grid_postprocess_is_physically_large(
        accumulator_voxels,
    ) or not relion_functions._large_grid_postprocess_single_precision_enabled(
        reconstruction_voxels,
    ):
        return Ft_ctf, Ft_y

    def _is_full(array):
        return tuple(array.shape) == accumulator_shape or (array.ndim == 1 and int(array.size) == accumulator_voxels)

    if not _is_full(Ft_ctf) or not _is_full(Ft_y):
        return Ft_ctf, Ft_y

    logger.info(
        "RELION giant-iFFT compact full-to-half repack: accumulator_shape=%s reconstruction_shape=%s",
        accumulator_shape,
        reconstruction_shape,
    )
    return (
        fourier_transform_utils.full_volume_to_half_volume(
            Ft_ctf,
            accumulator_shape,
        ).reshape(-1),
        fourier_transform_utils.full_volume_to_half_volume(
            Ft_y,
            accumulator_shape,
        ).reshape(-1),
    )


def _crop_relion_wiener_half_to_fftw_host(
    wiener_half,
    accumulator_shape,
    reconstruction_shape,
    relion_functions,
):
    """Crop a centered packed half-volume into raw FFTW order on the host."""

    accumulator_shape = tuple(int(s) for s in accumulator_shape)
    reconstruction_shape = tuple(int(s) for s in reconstruction_shape)
    accumulator_half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(
        accumulator_shape,
    )
    wiener_half = np.asarray(wiener_half).reshape(accumulator_half_shape)
    centered_axis_idx = relion_functions._relion_centered_axis_take_indices(
        accumulator_shape[0],
        reconstruction_shape[0],
    )
    raw_axis_idx = np.fft.ifftshift(centered_axis_idx)
    col_idx = np.arange(reconstruction_shape[-1] // 2 + 1, dtype=np.int32)
    return wiener_half[np.ix_(raw_axis_idx, raw_axis_idx, col_idx)]


def _pad_relion_wiener_half_to_fftw_host(
    wiener_half,
    accumulator_shape,
    reconstruction_shape,
    relion_functions,
):
    """Pad a centered packed Wiener half into raw FFTW order on the host.

    The host counterpart of recovar's ``_relion_pad_centered_half_fourier_to_fftw``
    (the padding branch of ``post_process_from_filter_v2``): the same support
    sphere is zeroed and the same placement is written, so the result equals the
    device pad exactly.
    """

    accumulator_shape = tuple(int(s) for s in accumulator_shape)
    reconstruction_shape = tuple(int(s) for s in reconstruction_shape)
    old_dim, new_dim = accumulator_shape[0], reconstruction_shape[0]
    if len(set(accumulator_shape)) != 1 or len(set(reconstruction_shape)) != 1 or new_dim <= old_dim:
        raise ValueError(
            f"host Wiener padding needs cubic shapes with a larger target, got {accumulator_shape} -> "
            f"{reconstruction_shape}"
        )
    old_half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(accumulator_shape)
    new_half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(reconstruction_shape)
    wiener_half = np.asarray(wiener_half).reshape(old_half_shape)
    freq = relion_functions._relion_centered_axis_fftw_frequencies(old_dim).astype(np.int64)
    raw_axis_idx = np.where(freq >= 0, freq, new_dim + freq)
    col_freq = np.arange(old_half_shape[-1], dtype=np.int64)
    max_r2 = int(old_half_shape[-1] - 1) ** 2
    out = np.zeros(new_half_shape, dtype=wiener_half.dtype)
    freq2 = freq * freq
    for i0 in range(old_dim):
        # One plane at a time keeps the support mask small (old_dim^2 x cols).
        support = (freq2[i0] + freq2[:, None] + (col_freq * col_freq)[None, :]) <= max_r2
        plane = np.where(support, wiener_half[i0], np.zeros((), dtype=wiener_half.dtype))
        out[raw_axis_idx[i0][None, None], raw_axis_idx[:, None], col_freq[None, :]] = plane
    return out


def _device_array_is_deleted(value):
    """Return JAX's deletion state when the device-array API exposes it."""

    is_deleted = getattr(value, "is_deleted", None)
    return bool(is_deleted()) if callable(is_deleted) else None


def _finish_host_staged_reconstruction(result, *accumulators):
    """Finish a host-staged reconstruction before dispatching the next half.

    Large RELION accumulators are moved to NumPy before reconstruction so the
    device only needs one half's inputs and FFT workspace at a time.  JAX
    dispatch is asynchronous, so wait here before the loop starts the other
    half; otherwise both padded FFT workspaces can overlap despite the host
    staging boundary.
    """

    if any(isinstance(accumulator, np.ndarray) for accumulator in accumulators):
        result.block_until_ready()
        gc.collect()
    return result


_LARGE_RELION_SOLVENT_MASK_COORDINATE_BYTES_LIMIT = 2 * 1024**3


def _relion_solvent_mask_unfused_coordinate_bytes(volume_shape) -> int:
    """Estimate the promoted coordinate stack used by ``raised_cosine_mask``."""

    return math.prod(int(size) for size in volume_shape) * 3 * np.dtype(np.float64).itemsize


def _large_relion_solvent_mask_uses_compiled_builder(volume_shape) -> bool:
    """Return whether the unfused solvent-mask coordinate stack is too large."""

    return (
        _relion_solvent_mask_unfused_coordinate_bytes(volume_shape) > _LARGE_RELION_SOLVENT_MASK_COORDINATE_BYTES_LIMIT
    )


@functools.cache
def _compiled_relion_solvent_mask(volume_shape, *, dtype=None):
    """Cache a shape-specialized builder, without retaining a mask array."""

    volume_shape = tuple(int(size) for size in volume_shape)

    @jax.jit
    def build(radius, radius_p, offset):
        return mask.raised_cosine_mask(
            volume_shape,
            radius=radius,
            radius_p=radius_p,
            offset=offset,
            dtype=dtype,
        )

    return build


def _make_relion_solvent_mask(volume_shape, *, radius, radius_p, dtype):
    """Build a RELION solvent mask, centred, without materializing a giant coordinate stack."""

    volume_shape = tuple(int(size) for size in volume_shape)
    if not _large_relion_solvent_mask_uses_compiled_builder(volume_shape):
        return mask.raised_cosine_mask(
            volume_shape,
            radius=radius,
            radius_p=radius_p,
            offset=jnp.zeros(3),
            dtype=dtype,
        )

    estimated_bytes = _relion_solvent_mask_unfused_coordinate_bytes(volume_shape)
    logger.info(
        "RELION box-scale solvent mask fused construction: shape=%s estimated_unfused_coordinate_bytes=%d",
        volume_shape,
        estimated_bytes,
    )
    solvent_mask = _compiled_relion_solvent_mask(volume_shape, dtype=dtype)(
        radius,
        radius_p,
        jnp.zeros(3),
    )
    solvent_mask.block_until_ready()
    logger.info(
        "RELION box-scale solvent mask ready: shape=%s dtype=%s",
        volume_shape,
        solvent_mask.dtype,
    )
    return solvent_mask


def _apply_relion_solvent_flatten_k1(
    volume_ft_flat,
    solvent_mask,
    volume_shape,
):
    """Apply the K=1 solvent mask and host-stage box-scale FFT results."""

    # One program (_flatten_volume): the eager statements held four box-size shift copies and transforms at once, and
    # at box 448 on a 16 GB card the fifth could not be placed (relax#49). The device CTF row caches yield if the
    # device cannot hand out the program's working set now, as before the device inverse FFT.
    relion_ctf.ensure_device_headroom(int(_DEVICE_FLATTEN_MAPS * np.prod(volume_shape) * 8))
    flattened = _flatten_volume(volume_ft_flat, solvent_mask, volume_shape=tuple(int(n) for n in volume_shape))
    if not _large_relion_solvent_mask_uses_compiled_builder(volume_shape):
        return flattened

    # JAX dispatch is asynchronous.  At box 800, even after the first half's
    # real-space volume and mask are released, retaining its complex128 FFT
    # output leaves too little room for the second normalized FFT allocation.
    # Finish the exact existing FFT, copy its completed bits to host memory,
    # and release all three device buffers before starting the next half.  The
    # transform arithmetic, normalization, dtype, and flattened shape remain
    # unchanged; only the storage owner crosses the device/host boundary.
    flattened.block_until_ready()
    flattened_host = np.array(jax.device_get(flattened), copy=True, order="C")
    regularization_relion.delete_device_array(flattened)
    regularization_relion.delete_device_array(solvent_mask)
    gc.collect()
    return flattened_host
