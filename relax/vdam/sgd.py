"""Opt-in curvature-normalized momentum SGD for InitialModel: the scalar per-class curvature step and
its discounted observation-space residual noise.

The E-step supplies the posterior-weighted residual backprojection and its
diagonal Fourier curvature on the same class/pose support. The maximum active
curvature is one scalar for the entire class, as in cryoSPARC Supplement Eq. 9.
The Fourier-to-map transform uses the RELION InitialModel M-step convention;
the map and momentum remain float32 on device. No VDAM half-difference or
voxelwise inverse-curvature operation enters this update.

The cryoSPARC Supplement gives a decaying residual estimate and white/inflated
priors. Here the shared RELION E-step's posterior-weighted residual sums are
converted to per-component RELION variance before accumulating. Only observed
Fourier shells accrue data count; the scalar baseline is measured from raw
particle corners, without GT or model images.

See ``docs/math/momentum_sgd.md`` for the exact surrogate objective, its units and the
observation-space noise adaptation.
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

from relax.fourier.half_spectrum import _host_half_spectrum_plan
from relax.reconstruction.noise import normalize_wsum_to_sigma2_noise
from relax.relion.project import gridding_correct_volume_real
from relax.relion.projector_setup import setup_relion_projector_uncorrected, swap_relion_volume_layout
from relax.relion.vdam_mstep import _pad_moment
from relax.vdam.state import InitialModelState, VdamAccumulator

MOMENTUM = 0.9


def _inverse_bpref_fourier(volume_half, *, box_size, padding_factor):
    """Inverse the shared RELION centered half-volume to a real map."""
    fft_size = int(box_size) * int(padding_factor)
    capacity = fft_size + 3
    fft_half = relion_window_centered_half_fourier(volume_half, (capacity,) * 3, (fft_size,) * 3)
    dc = (fft_size // 2, fft_size // 2, 0)
    fft_half = fft_half.at[dc].set(jnp.real(fft_half[dc]) + 0.0j)
    real = ftu.get_idft3_real(fft_half, (fft_size,) * 3, norm="forward")
    start = (fft_size - int(box_size)) // 2
    return real[start : start + box_size, start : start + box_size, start : start + box_size] / float(
        padding_factor**3 * box_size
    )


def _bandlimit_real_map(volume_relion, radius, *, box_size, padding_factor):
    """Project a map onto the same Fourier support used by its BPref."""
    projector, _ = setup_relion_projector_uncorrected(
        volume_relion,
        radius,
        box_size=box_size,
        padding_factor=padding_factor,
        compute_dtype=jnp.float32,
    )
    return _inverse_bpref_fourier(projector, box_size=box_size, padding_factor=padding_factor)


@partial(jax.jit, static_argnames=("box_size", "padding_factor"))
def _class_step(reference, previous_update, residual, curvature, radius, learning_rate, *, box_size, padding_factor):
    """Return one float32 map step in RECOVAR axes and its scalar diagnostics."""
    fft_size = int(box_size) * int(padding_factor)
    capacity = fft_size + 3
    coord = jnp.arange(capacity, dtype=jnp.int32) - capacity // 2
    x = jnp.arange(capacity // 2 + 1, dtype=jnp.int32)
    r2 = coord[:, None, None] ** 2 + coord[None, :, None] ** 2 + x[None, None, :] ** 2
    valid = r2 < (int(padding_factor) * radius) ** 2
    maximum = jnp.max(jnp.where(valid, curvature, 0.0))
    safe_maximum = jnp.where(maximum > 0.0, maximum, 1.0)
    scaled = jnp.where(valid, residual / safe_maximum, 0.0 + 0.0j)
    gradient_relion = _inverse_bpref_fourier(scaled, box_size=box_size, padding_factor=padding_factor)
    # The shared E-step projects sinc^-2 corrected maps. Its map-space adjoint
    # applies the same self-adjoint real-space correction to the raw BPref gradient.
    gradient_relion = gridding_correct_volume_real(gradient_relion, box_size, padding_factor)
    gradient_relion = _bandlimit_real_map(
        gradient_relion,
        radius,
        box_size=box_size,
        padding_factor=padding_factor,
    )
    gradient = swap_relion_volume_layout(gradient_relion, jnp.float32)
    previous_relion = swap_relion_volume_layout(previous_update, jnp.float32)
    previous_band = swap_relion_volume_layout(
        _bandlimit_real_map(previous_relion, radius, box_size=box_size, padding_factor=padding_factor), jnp.float32
    )
    reference_relion = swap_relion_volume_layout(reference, jnp.float32)
    reference_band = swap_relion_volume_layout(
        _bandlimit_real_map(reference_relion, radius, box_size=box_size, padding_factor=padding_factor), jnp.float32
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
    """Apply the paper-described scalar-curvature momentum update to all K maps."""
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
    if radius < 1 or radius > int(state.box_size) // 2:
        raise ValueError("SGD current Fourier radius is outside the model box")
    references, updates = [], []
    curvature_max, gradient_norm, step_norm = [], [], []
    for k in range(state.K):
        residual, curvature = _pooled_class_accumulators(accumulators, state.K, k)
        capacity = int(padding_factor) * int(state.box_size) + 3
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
            box_size=int(state.box_size),
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


GAMMA = 0.9999
WHITE_PRIOR_COUNT = 50.0
INFLATED_PRIOR_COUNT = 2500.0
INFLATED_PRIOR_VARIANCE = 8.0


def corner_white_sigma2(images: np.ndarray, image_mask: np.ndarray, *, image_multiplier: float = 1.0) -> np.ndarray:
    """Estimate masked-scoring Fourier noise from raw pixel corners.

    The particle pixels are modeled as independent white noise with variance
    measured outside the mask. The exact linear background-fill operator then
    maps that variance to each Fourier coefficient before radial averaging.
    """
    images = np.asarray(images, dtype=np.float64)
    if not np.isfinite(image_multiplier) or image_multiplier == 0.0:
        raise ValueError("image_multiplier must be finite and nonzero")
    if images.ndim != 3 or images.shape[1] != images.shape[2] or images.shape[0] < 1:
        raise ValueError("corner noise needs a nonempty stack of square raw images")
    n = int(images.shape[1])
    image_mask = np.asarray(image_mask, dtype=np.float64)
    if (
        image_mask.shape != (n, n)
        or not np.all(np.isfinite(image_mask))
        or np.any((image_mask < 0.0) | (image_mask > 1.0))
    ):
        raise ValueError("image_mask must be a finite square mask in [0, 1]")
    corners = image_mask <= 1e-6
    if np.count_nonzero(corners) < 4:
        raise ValueError("particle support leaves fewer than four background pixels")
    background = images[:, corners]
    # Each image may have a different DC offset; remove it before white-power estimation.
    pixel_variance = np.mean((background - background.mean(axis=1, keepdims=True)) ** 2) * float(image_multiplier) ** 2
    m = np.fft.fftshift(image_mask)
    w = 1.0 - m
    W = float(w.sum())
    if W <= 0.0:
        raise ValueError("scoring mask has no background pixels")
    w_fft = np.fft.rfft2(w)
    c = w_fft / W
    mw_fft = np.fft.rfft2(m * w)
    transfer_power = float(np.sum(m * m)) + np.abs(c) ** 2 * float(np.sum(w * w)) + 2.0 * np.real(np.conj(c) * mw_fft)
    transfer_power = np.fft.fftshift(transfer_power, axes=(0,))
    n_shells = n // 2 + 1
    shell_ids = _host_half_spectrum_plan((n, n)).relion_noise_shell_indices.reshape(n, n // 2 + 1)
    selected = shell_ids < n_shells
    power_sum = np.bincount(
        shell_ids[selected].reshape(-1), weights=transfer_power[selected].reshape(-1), minlength=n_shells
    )
    count = np.bincount(shell_ids[selected].reshape(-1), minlength=n_shells)
    sigma2 = float(pixel_variance) * power_sum / (2.0 * float(n**4) * np.maximum(count, 1))
    if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0.0):
        raise ValueError("corner white variance must be positive and finite")
    return sigma2


def initialize_sgd_noise(state: InitialModelState, corner_sigma2: np.ndarray) -> InitialModelState:
    """Start at the white-plus-inflated prior mean before the first E-step."""
    if state.sigma2_noise.shape[0] != 1:
        raise ValueError("SGD initial noise currently requires one optics group")
    baseline = np.asarray(corner_sigma2, dtype=np.float64)
    if baseline.shape != (state.sigma2_noise.shape[1],) or not np.all(np.isfinite(baseline)) or np.any(baseline <= 0.0):
        raise ValueError("corner_sigma2 must be positive, finite and match the shell count")
    prior = (WHITE_PRIOR_COUNT + INFLATED_PRIOR_COUNT * INFLATED_PRIOR_VARIANCE) / (
        WHITE_PRIOR_COUNT + INFLATED_PRIOR_COUNT
    )
    return replace(
        state,
        sigma2_noise=(prior * baseline)[None, :],
        sgd_corner_sigma2=baseline.copy(),
        sgd_noise_sum=np.zeros(state.sigma2_noise.shape[1], dtype=np.float64),
        sgd_noise_count=np.zeros(state.sigma2_noise.shape[1], dtype=np.float64),
    )


def update_sgd_noise(state: InitialModelState, meta: dict) -> InitialModelState:
    """Update discounted residual numerator/count plus persistent/decaying priors."""
    if state.sgd_corner_sigma2 is None or state.sgd_noise_sum is None or state.sgd_noise_count is None:
        raise ValueError("SGD noise state was not initialized")
    required = ("wsum_sigma2_noise", "wsum_img_power", "noise_sumw")
    if any(meta.get(key) is None for key in required):
        raise ValueError("SGD E-step did not report residual noise sums and count")
    residual = np.asarray(meta["wsum_sigma2_noise"], dtype=np.float64)
    image_power = np.asarray(meta["wsum_img_power"], dtype=np.float64)
    mass = float(meta["noise_sumw"])
    shells = int(state.box_size) // 2 + 1
    if residual.shape != (shells,) or image_power.shape != (shells,):
        raise ValueError("SGD residual noise has the wrong Fourier shell shape")
    if not (np.all(np.isfinite(residual)) and np.all(np.isfinite(image_power)) and np.isfinite(mass) and mass > 0.0):
        raise ValueError("SGD residual noise sums/count must be finite and count positive")
    batch_noise = np.asarray(
        normalize_wsum_to_sigma2_noise(residual, image_power, mass, (state.box_size,) * 2),
        dtype=np.float64,
    ) / float(state.box_size**4)
    active = np.arange(shells) <= int(state.current_size) // 2
    batch_noise = np.where(active, batch_noise, 0.0)
    if not np.all(np.isfinite(batch_noise)) or np.any(batch_noise < 0.0):
        raise ValueError("SGD batch noise must be finite and non-negative")
    numerator = GAMMA * np.asarray(state.sgd_noise_sum, dtype=np.float64) + mass * batch_noise
    count = GAMMA * np.asarray(state.sgd_noise_count, dtype=np.float64) + mass * active
    inflated_weight = INFLATED_PRIOR_COUNT * GAMMA ** int(state.iter)
    baseline = np.asarray(state.sgd_corner_sigma2, dtype=np.float64)
    sigma2 = (numerator + (WHITE_PRIOR_COUNT + INFLATED_PRIOR_VARIANCE * inflated_weight) * baseline) / (
        count + WHITE_PRIOR_COUNT + inflated_weight
    )
    if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0.0):
        raise ValueError("SGD updated noise must be positive and finite")
    meta["sgd_noise_data_count_by_shell"] = count.tolist()
    meta["sgd_noise_white_prior_count"] = WHITE_PRIOR_COUNT
    meta["sgd_noise_inflated_prior_count"] = float(inflated_weight)
    meta["sgd_noise_prior_variance_by_shell"] = baseline.tolist()
    return replace(
        state,
        sgd_noise_sum=numerator,
        sgd_noise_count=count,
        sigma2_noise=sigma2[None, :],
    )
