"""Validate and select inputs for dense and local K-class EM.

Class scheduling and engine dispatch stay in ``k_class``. These helpers own
class-axis validation, shared/per-class selection and local prior layouts;
they do not import execution engines or change scoring precision.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _class_log_priors(n_classes: int, class_log_priors) -> np.ndarray:
    if class_log_priors is None:
        return np.full(n_classes, -np.log(float(n_classes)), dtype=np.float64)
    priors = np.asarray(class_log_priors, dtype=np.float64)
    if priors.shape != (n_classes,):
        raise ValueError(f"class_log_priors must have shape ({n_classes},), got {priors.shape}")
    if not np.all(np.isfinite(priors)):
        raise ValueError("class_log_priors must be finite")
    return priors


def _as_class_means(means) -> jax.Array:
    means_array = jnp.asarray(means)
    if means_array.ndim != 2:
        raise ValueError(f"means must have shape (n_classes, volume_size), got {means_array.shape}")
    if int(means_array.shape[0]) < 1:
        raise ValueError("means must contain at least one class")
    return means_array


def _select_class_value(value, class_index: int, n_classes: int):
    value_array = jnp.asarray(value)
    if value_array.ndim >= 2 and int(value_array.shape[0]) == n_classes:
        return value_array[class_index]
    return value


def _select_projector_half_for_class(value, class_index: int, n_classes: int):
    """Select one RELION projector before transferring it to the device.

    Production projector slabs are NumPy arrays.  Preserve a host view for
    both singleton reshape and K-class indexing so only the selected 3-D slab
    reaches ``jnp.asarray`` in its consumer.  A traced JAX value may reshape
    inside its enclosing compilation.  Reject eager 4-D JAX arrays because
    even a logically shape-only reshape can allocate a second device buffer.
    """
    if value is None:
        return None
    value_array = value if isinstance(value, (np.ndarray, jax.Array, jax.core.Tracer)) else np.asarray(value)
    if value_array.ndim >= 4 and int(value_array.shape[0]) == int(n_classes):
        if isinstance(value_array, jax.Array) and not isinstance(value_array, jax.core.Tracer):
            raise ValueError(
                "RELION projector class selection must occur before eager "
                "device transfer; pass the NumPy host array or select inside "
                "an enclosing jax.jit trace",
            )
        if int(n_classes) == 1:
            if int(class_index) != 0:
                raise IndexError("a singleton RELION projector only has class index 0")
            if isinstance(value_array, jax.core.Tracer):
                return jnp.reshape(value_array, value_array.shape[1:])
            return np.reshape(value_array, value_array.shape[1:])
        return value_array[int(class_index)]
    return value


def _select_required_class_value(value, class_index: int, n_classes: int, name: str):
    value_array = jnp.asarray(value)
    if value_array.ndim < 2 or int(value_array.shape[0]) != n_classes:
        raise ValueError(f"{name} must have leading class axis of length {n_classes}, got {value_array.shape}")
    return value_array[class_index]




def seed_iteration_first_class(means, class_rotation_log_prior=None):
    """Check that a seed iteration's classes are copies of one reference; return its first class's slices.

    RELION's Class3D from one reference (``do_generate_seeds``, ml_model.cpp:1007-1010) copies it to every class
    and, in the first iteration, scores each particle against one random class only (ml_optimiser.cpp:4880-4898).
    Every copy gives the same diff2, so pass 1 scores the first class alone; its ``log pdf_class`` shifts every
    weight of the particle equally, so the significance cut and Pmax are the random class's
    (:func:`seed_iteration_supports`). Returns ``(means[:1], class_rotation_log_prior[:1] or None)``.
    """

    host = np.asarray(means)
    if host.ndim < 2 or not all(np.array_equal(host[k], host[0]) for k in range(1, host.shape[0])):
        raise ValueError("a seed iteration's classes must be copies of one reference")
    if class_rotation_log_prior is not None:
        prior = np.asarray(class_rotation_log_prior)
        if prior.ndim == 2:
            if not all(np.array_equal(prior[k], prior[0]) for k in range(1, prior.shape[0])):
                raise ValueError("a seed iteration's classes must share one direction prior")
            class_rotation_log_prior = prior[:1]
    return means[:1], class_rotation_log_prior


def seed_iteration_supports(first_class_supports, unit_seed_classes, n_classes: int):
    """Per-class pass-1 supports of a seed iteration: each unit's first-class support in its random class only.

    ``first_class_supports[u]`` is unit ``u``'s coarse support scored against the first class
    (:func:`seed_iteration_first_class`) and ``unit_seed_classes[u]`` its random class (0-based,
    ``relax.relion.input_particle_table.relion_class3d_seed_classes``); the other classes get no candidates, as
    RELION scores none of them.
    """

    seeds = np.asarray(unit_seed_classes, dtype=np.int64).reshape(-1)
    if seeds.size != len(first_class_supports) or np.any((seeds < 0) | (seeds >= int(n_classes))):
        raise ValueError("a seed iteration gives every unit one class in range")
    from relax.sparse_pass2.resident_significance import DeviceCompactedSignificantSamples, host_support_rows

    if isinstance(first_class_supports, DeviceCompactedSignificantSamples):
        classes = []
        for k in range(int(n_classes)):
            csr = first_class_supports.csr.select_images(seeds == k)
            classes.append(DeviceCompactedSignificantSamples(host_support_rows(csr), csr=csr))
        return classes
    empty = np.zeros(0, dtype=np.int32)
    return [
        [support if seeds[u] == k else empty for u, support in enumerate(first_class_supports)]
        for k in range(int(n_classes))
    ]
