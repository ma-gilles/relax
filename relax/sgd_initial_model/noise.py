"""Discounted observation-space residual noise for opt-in InitialModel SGD.

The cryoSPARC Supplement gives a decaying residual estimate and white/inflated
priors. Here the shared RELION E-step's posterior-weighted residual sums are
converted to per-component RELION variance before accumulating. Only observed
Fourier shells accrue data count; the scalar baseline is measured from raw
particle corners, without GT or model images.

See ``docs/math/cryosparc_sgd.md`` for the observation-space adaptation.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from relax.reconstruction.noise_relion import normalize_wsum_to_sigma2_noise
from relax.helpers.half_spectrum import _host_half_spectrum_plan
from relax.vdam.state import InitialModelState

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
    if image_mask.shape != (n, n) or not np.all(np.isfinite(image_mask)) or np.any((image_mask < 0.0) | (image_mask > 1.0)):
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
    transfer_power = float(np.sum(m * m)) + np.abs(c) ** 2 * float(np.sum(w * w)) + 2.0 * np.real(
        np.conj(c) * mw_fft
    )
    transfer_power = np.fft.fftshift(transfer_power, axes=(0,))
    n_shells = n // 2 + 1
    shell_ids = _host_half_spectrum_plan((n, n)).relion_noise_shell_indices.reshape(n, n // 2 + 1)
    selected = shell_ids < n_shells
    power_sum = np.bincount(shell_ids[selected].reshape(-1), weights=transfer_power[selected].reshape(-1), minlength=n_shells)
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
    shells = int(state.ori_size) // 2 + 1
    if residual.shape != (shells,) or image_power.shape != (shells,):
        raise ValueError("SGD residual noise has the wrong Fourier shell shape")
    if not (np.all(np.isfinite(residual)) and np.all(np.isfinite(image_power)) and np.isfinite(mass) and mass > 0.0):
        raise ValueError("SGD residual noise sums/count must be finite and count positive")
    batch_noise = np.asarray(
        normalize_wsum_to_sigma2_noise(residual, image_power, mass, (state.ori_size,) * 2),
        dtype=np.float64,
    ) / float(state.ori_size**4)
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
