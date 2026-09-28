"""Scalar per-class curvature step on the shared InitialModel residual BPref.

The E-step supplies the posterior-weighted residual backprojection and its
diagonal Fourier curvature on the same class/pose support. The maximum active
curvature is one scalar for the entire class, as in cryoSPARC Supplement Eq. 9.
The Fourier-to-map transform uses the RELION InitialModel M-step convention;
the map and momentum remain float32 on device. No VDAM half-difference or
voxelwise inverse-curvature operation enters this update.

See ``docs/math/cryosparc_sgd.md`` for the exact surrogate objective and units.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from recovar.core import fourier_transform_utils as ftu
from recovar.core import mask
from recovar.reconstruction.relion_functions import relion_window_centered_half_fourier

from relax.relion.relion_project import gridding_correct_volume_real
from relax.relion.relion_projector_setup import setup_relion_projector_uncorrected, swap_relion_volume_layout
from relax.relion.relion_vdam_mstep import _pad_moment
from relax.vdam.state import InitialModelState, VdamAccumulator

MOMENTUM = 0.9


def _inverse_bpref_fourier(volume_half, *, ori_size, padding_factor):
    """Inverse the shared RELION centered half-volume to a real map."""
    fft_size = int(ori_size) * int(padding_factor)
    capacity = fft_size + 3
    fft_half = relion_window_centered_half_fourier(volume_half, (capacity,) * 3, (fft_size,) * 3)
    dc = (fft_size // 2, fft_size // 2, 0)
    fft_half = fft_half.at[dc].set(jnp.real(fft_half[dc]) + 0.0j)
    real = ftu.get_idft3_real(fft_half, (fft_size,) * 3, norm="forward")
    start = (fft_size - int(ori_size)) // 2
    return real[start : start + ori_size, start : start + ori_size, start : start + ori_size] / float(
        padding_factor**3 * ori_size
    )


def _bandlimit_real_map(volume_relion, radius, *, ori_size, padding_factor):
    """Project a map onto the same Fourier support used by its BPref."""
    projector, _ = setup_relion_projector_uncorrected(
        volume_relion,
        radius,
        ori_size=ori_size,
        padding_factor=padding_factor,
        compute_dtype=jnp.float32,
    )
    return _inverse_bpref_fourier(projector, ori_size=ori_size, padding_factor=padding_factor)


@partial(jax.jit, static_argnames=("ori_size", "padding_factor"))
def _class_step(reference, previous_update, residual, curvature, radius, learning_rate, *, ori_size, padding_factor):
    """Return one float32 map step in RECOVAR axes and its scalar diagnostics."""
    fft_size = int(ori_size) * int(padding_factor)
    capacity = fft_size + 3
    coord = jnp.arange(capacity, dtype=jnp.int32) - capacity // 2
    x = jnp.arange(capacity // 2 + 1, dtype=jnp.int32)
    r2 = coord[:, None, None] ** 2 + coord[None, :, None] ** 2 + x[None, None, :] ** 2
    valid = r2 < (int(padding_factor) * radius) ** 2
    maximum = jnp.max(jnp.where(valid, curvature, 0.0))
    safe_maximum = jnp.where(maximum > 0.0, maximum, 1.0)
    scaled = jnp.where(valid, residual / safe_maximum, 0.0 + 0.0j)
    gradient_relion = _inverse_bpref_fourier(scaled, ori_size=ori_size, padding_factor=padding_factor)
    # The shared E-step projects sinc^-2 corrected maps. Its map-space adjoint
    # applies the same self-adjoint real-space correction to the raw BPref gradient.
    gradient_relion = gridding_correct_volume_real(gradient_relion, ori_size, padding_factor)
    gradient_relion = _bandlimit_real_map(
        gradient_relion,
        radius,
        ori_size=ori_size,
        padding_factor=padding_factor,
    )
    gradient = swap_relion_volume_layout(gradient_relion, jnp.float32)
    previous_relion = swap_relion_volume_layout(previous_update, jnp.float32)
    previous_band = swap_relion_volume_layout(
        _bandlimit_real_map(previous_relion, radius, ori_size=ori_size, padding_factor=padding_factor), jnp.float32
    )
    reference_relion = swap_relion_volume_layout(reference, jnp.float32)
    reference_band = swap_relion_volume_layout(
        _bandlimit_real_map(reference_relion, radius, ori_size=ori_size, padding_factor=padding_factor), jnp.float32
    )
    update = MOMENTUM * previous_band + (1.0 - MOMENTUM) * learning_rate * gradient
    update = jnp.where(maximum > 0.0, update, jnp.zeros_like(update))
    updated_reference, _ = mask.soft_mask_outside_map(reference_band + update)
    updated_reference = jnp.where(maximum > 0.0, updated_reference, reference)
    return updated_reference, update, maximum, jnp.linalg.norm(gradient), jnp.linalg.norm(update)


def _pooled_class_accumulators(accumulators: list[VdamAccumulator], K: int, class_idx: int):
    """Sum both pseudo-halfsets before using the joint class gradient/curvature."""
    first = accumulators[class_idx]
    second = accumulators[K + class_idx]
    if (first.class_idx, first.halfset_idx, second.class_idx, second.halfset_idx) != (class_idx, 0, class_idx, 1):
        raise ValueError("SGD requires halfset-major K-class BPref accumulators")
    if first.data.shape != second.data.shape or first.weight.shape != second.weight.shape:
        raise ValueError("SGD halfset BPref shapes differ")
    residual = jnp.asarray(first.data, jnp.complex64) + jnp.asarray(second.data, jnp.complex64)
    curvature = jnp.asarray(first.weight, jnp.float32) + jnp.asarray(second.weight, jnp.float32)
    if residual.shape != curvature.shape:
        raise ValueError("SGD pooled residual and curvature shapes differ")
    return residual, curvature


def sgd_m_step(
    state: InitialModelState,
    accumulators: list[VdamAccumulator],
    *,
    learning_rate: float,
    padding_factor: int,
    meta: dict,
) -> InitialModelState:
    """Apply the cryoSPARC-style scalar-curvature momentum update to all K maps."""
    if not state.pseudo_halfsets or len(accumulators) != 2 * state.K:
        raise ValueError("SGD expects two pseudo-halfset accumulators per class")
    if state.Iref.dtype != np.dtype(np.float32):
        raise ValueError("production SGD references must be float32")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("SGD learning_rate must be positive and finite")
    previous = state.sgd_previous_update
    if previous is None:
        previous = jnp.zeros_like(state.Iref, dtype=jnp.float32)
    if previous.shape != state.Iref.shape or previous.dtype != np.dtype(np.float32):
        raise ValueError("SGD previous update shape/dtype does not match references")
    radius = int(state.current_size) // 2
    if radius < 1 or radius > int(state.ori_size) // 2:
        raise ValueError("SGD current Fourier radius is outside the model box")
    references, updates = [], []
    curvature_max, gradient_norm, step_norm = [], [], []
    for k in range(state.K):
        residual, curvature = _pooled_class_accumulators(accumulators, state.K, k)
        capacity = int(padding_factor) * int(state.ori_size) + 3
        if residual.ndim != 3 or residual.shape[0] > capacity or residual.shape[1] != residual.shape[0]:
            raise ValueError("SGD BPref has invalid capacity")
        if residual.shape[0] // 2 < int(padding_factor) * radius:
            raise ValueError("SGD BPref does not cover the current Fourier radius")
        residual = _pad_moment(residual, capacity)
        curvature = _pad_moment(curvature, capacity)
        if not bool(jnp.all(jnp.isfinite(residual))) or not bool(jnp.all(jnp.isfinite(curvature))):
            raise ValueError("SGD residual/curvature must be finite")
        if bool(jnp.any(curvature < 0.0)):
            raise ValueError("SGD curvature must be non-negative")
        reference, update, maximum, grad_norm, delta_norm = _class_step(
            jnp.asarray(state.Iref[k], jnp.float32),
            jnp.asarray(previous[k], jnp.float32),
            residual,
            curvature,
            np.int32(radius),
            np.float32(learning_rate),
            ori_size=int(state.ori_size),
            padding_factor=int(padding_factor),
        )
        if not bool(jnp.all(jnp.isfinite(reference))) or not bool(jnp.all(jnp.isfinite(update))):
            raise ValueError("SGD reference/update became non-finite")
        references.append(reference)
        updates.append(update)
        curvature_max.append(float(maximum))
        gradient_norm.append(float(grad_norm))
        step_norm.append(float(delta_norm))
    meta["sgd_max_curvature_by_class"] = curvature_max
    meta["sgd_normalized_gradient_norm_by_class"] = gradient_norm
    meta["sgd_step_norm_by_class"] = step_norm
    meta["sgd_learning_rate"] = float(learning_rate)
    return replace(state, Iref=jnp.stack(references), sgd_previous_update=jnp.stack(updates))
