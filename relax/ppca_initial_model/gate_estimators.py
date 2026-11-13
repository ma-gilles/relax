"""Gate study: shell statistics for a model-based VDAM gate (docs/math/vdam_gate_study.md, arms C and D).

Per Fourier voxel and pseudo-half h, a batch gives the right-hand side B_h and the weight F_h (one channel; the
PPCA channels use the diagonal of their block). With a regularizer lambda > 0, the half's regularized solution is

    w_h = B_h / (F_h + lambda) = A_h theta_hat_h,   A_h = F_h / (F_h + lambda),

computed from the residual R_h = B_h - F_h v of the shared model v as w_h = R_h / (F_h + lambda) + A_h v. If
theta_hat_h = s + e_h with independent noises of variance kappa / F_h, then E[w_0 conj(w_1)] = A_0 A_1 |s|^2 and
E|w_0 - w_1|^2 = (A_0 - A_1)^2 |s|^2 + kappa (F_0 / (F_0 + lambda)^2 + F_1 / (F_1 + lambda)^2). Summed over a shell
(and over batches), the ratios below are consistent estimators of the shell's signal power P and of the noise
calibration kappa (1 when the model's 1 / F is the noise variance). A single batch's F_h is never inverted on its own:
lambda keeps every voxel finite, and thin coverage only lowers A_h.
"""

import numpy as np


def regularized_halves(residual, weight, model, regularizer):
    """``w_h`` and ``A_h`` of both halves: ``residual`` [2, F] (R_h), ``weight`` [2, F] (F_h >= 0), ``model`` [F]."""
    residual = np.asarray(residual)
    weight = np.asarray(weight, np.float64)
    denominator = weight + regularizer
    gain = weight / denominator
    return residual / denominator + gain * np.asarray(model)[None], gain


def shell_sums(residual, weight, model, regularizer, shells, nshells):
    """Per-shell sums of one batch: [nshells, 5] = (sum Re w0 conj w1, sum A0 A1, sum |w0 - w1|^2,
    sum (A0 - A1)^2, sum F0/(F0+l)^2 + F1/(F1+l)^2). Accumulate them over batches, then call ``signal_and_noise``."""
    w, a = regularized_halves(residual, weight, model, regularizer)
    weight = np.asarray(weight, np.float64)
    noise_gain = weight / (weight + regularizer) ** 2
    columns = (
        np.real(w[0] * np.conj(w[1])),
        a[0] * a[1],
        np.abs(w[0] - w[1]) ** 2,
        (a[0] - a[1]) ** 2,
        noise_gain[0] + noise_gain[1],
    )
    return np.stack([np.bincount(shells, c, nshells) for c in columns], axis=-1)


def signal_and_noise(sums, floor=0.0):
    """Shell signal power P and noise calibration kappa from accumulated ``shell_sums``; P is clamped at ``floor``
    (returned mask: where the clamp acted) and kappa at 0."""
    sums = np.asarray(sums, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        signal = np.where(sums[..., 1] > 0, sums[..., 0] / sums[..., 1], 0.0)
        clamped = signal < floor
        signal = np.maximum(signal, floor)
        kappa = np.where(sums[..., 4] > 0, (sums[..., 2] - signal * sums[..., 3]) / sums[..., 4], np.nan)
    return signal, np.maximum(kappa, 0.0), clamped
