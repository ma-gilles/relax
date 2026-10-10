"""The reference state a refinement carries between iterations: each half's maps and tau2
(``ReferenceModel``), the class mixture, and how both start (from the start-up arrays or a snapshot)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from relax.helpers.orientation_priors import (
    class_weights_from_direction_prior,
)


@dataclass
class ReferenceModel:
    """Half/class Fourier maps and the tau2 volumes used by their next expectation.

    Maps stay in the existing flat centered Fourier layout. K1 map slots are
    cleared before reconstruction; Class3D slots share one class stack. Tau2
    retains the RECOVAR frame scale. ``tau2_per_half`` is each half's own tau2 when the halves score
    per half (K=1 only), and None when both score with the shared ``tau2``.
    """

    maps: list
    tau2: object
    tau2_per_half: list | None

    def half_tau2(self, half_index: int):
        """The tau2 half ``half_index`` scores with."""
        return self.tau2 if self.tau2_per_half is None else self.tau2_per_half[half_index]


def initialize_reference_model(half_maps, initial_mean_variance, *, use_per_half_mean_variance, dtype, log):
    """K=1: attach shared/per-half tau2 to the already normalized references; a per-half pair's mean is in the
    scoring ``dtype``."""
    if use_per_half_mean_variance:
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
        mean_variance_per_half = None
    return ReferenceModel(maps=half_maps, tau2=mean_variance, tau2_per_half=mean_variance_per_half)


def initialize_class_reference_model(half_maps, initial_mean_variance, *, use_per_half_mean_variance):
    """Class3D: the already normalized class stacks with the one shared tau2 (per-half tau2 is refused)."""
    if use_per_half_mean_variance:
        raise ValueError("per-half scoring tau2 is supported only for K=1")
    return ReferenceModel(maps=half_maps, tau2=initial_mean_variance, tau2_per_half=None)


def reference_model_from_snapshot(snapshot, volume_shape, *, dtype):
    """K=1: restore the checkpoint's half maps and the continuation's shared tau2 layout."""
    from relax.refinement.iteration_snapshot import tau2_mean_variance

    maps = [jnp.asarray(mean) for mean in snapshot.means]
    tau2 = tau2_mean_variance(snapshot, volume_shape, dtype=dtype)
    return ReferenceModel(maps=maps, tau2=tau2, tau2_per_half=None)


def class_reference_model_from_snapshot(snapshot, volume_shape, *, dtype):
    """Class3D: restore the checkpoint's class stack, one array both halves score against, and the shared
    tau2."""
    from relax.refinement.iteration_snapshot import tau2_mean_variance

    maps = [jnp.asarray(mean) for mean in snapshot.means]
    maps[1] = maps[0]
    tau2 = tau2_mean_variance(snapshot, volume_shape, dtype=dtype)
    return ReferenceModel(maps=maps, tau2=tau2, tau2_per_half=None)


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


def _updated_mean_variance_per_half(updated_mean_variance_per_half, *, use_per_half_mean_variance):
    """Each half's own scoring tau2 when per-half scoring is on; None (the shared tau2) otherwise."""

    if use_per_half_mean_variance:
        if len(updated_mean_variance_per_half) != 2:
            raise ValueError("per-half scoring tau2 update requires exactly two halves")
        return [
            jnp.asarray(updated_mean_variance_per_half[0]),
            jnp.asarray(updated_mean_variance_per_half[1]),
        ]
    return None


class ClassMixture(NamedTuple):
    """The class log priors scored against and the class weights they come from, both ``[K]`` float64."""

    log_priors: np.ndarray
    weights: np.ndarray


def class_mixture_from_weights(weights: np.ndarray) -> ClassMixture:
    """The mixture whose log priors are the logarithm of ``weights``."""

    return ClassMixture(np.log(weights), weights)


def _initialize_class_log_priors(n_classes: int, init_direction_prior=None) -> ClassMixture:
    """Return normalized log priors for the class axis and class weights: uniform, or (K > 1) the class weights
    a RELION direction prior's row sums imply."""
    if n_classes < 1:
        raise ValueError(f"n_classes must be >= 1, got {n_classes}")
    class_log_priors = np.full(n_classes, -np.log(float(n_classes)), dtype=np.float64)
    class_weights = np.exp(class_log_priors)
    if n_classes > 1 and init_direction_prior is not None:
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


def initial_half_references(init_volume, n_classes: int) -> list:
    """Each half's start-up reference in the loop's class layout, from the start-up ``HalfPair``: a flat
    reference for K=1, a ``(K, V)`` stack for Class3D (one flat reference is repeated for every class)."""

    def as_class_array(value):
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
            "a half's init_volume must be a flat reference or a per-class reference array compatible with "
            f"n_classes={n_classes}; got shape {tuple(arr.shape)}",
        )

    # Half-map updates replace list entries; immutable initial buffers can be shared.
    return list(init_volume.map(as_class_array))


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
