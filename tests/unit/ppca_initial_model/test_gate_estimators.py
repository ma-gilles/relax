"""The gate study's signal and noise estimators (arms C and D) on simulated Gaussian data with known P and noise."""

import numpy as np
import pytest

from relax.ppca_initial_model.gate_estimators import shell_sums, signal_and_noise

pytestmark = pytest.mark.unit


def _complex(rng, power, shape):
    return np.sqrt(power / 2) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def _simulate(rng, *, n_voxels, power, kappa, particles_per_half, batches, model_scale=0.3, regularizer=1.0):
    """Shell sums over ``batches`` batches: voxel weights F_h ~ Poisson-like (thin coverage on some voxels), half
    reconstructions theta_hat_h = s + e_h with Var e_h = kappa / F_h, a shared model v that is a shrunk noisy copy of s."""
    signal = _complex(rng, power, n_voxels)
    model = model_scale * signal + _complex(rng, 0.2 * power, n_voxels)
    shells = np.zeros(n_voxels, np.int64)
    total = np.zeros((1, 5))
    for _ in range(batches):
        weight = rng.gamma(2.0, particles_per_half / 2.0, (2, n_voxels)) * rng.choice([0.0, 1.0], (2, n_voxels), p=[0.05, 0.95])
        noise = _complex(rng, 1.0, (2, n_voxels)) * np.sqrt(kappa / np.maximum(weight, 1e-30))
        rhs = weight * (signal[None] + noise)  # B_h = F_h theta_hat_h
        residual = rhs - weight * model[None]
        total += shell_sums(residual, weight, model, regularizer, shells, 1)
    return total


@pytest.mark.parametrize("power, kappa", [(0.5, 1.0), (0.05, 0.3), (2.0, 2.5)])
def test_signal_and_noise_calibration_are_unbiased(power, kappa):
    rng = np.random.default_rng(7)
    sums = _simulate(rng, n_voxels=200_000, power=power, kappa=kappa, particles_per_half=2.0, batches=4)
    signal, kappa_hat, clamped = signal_and_noise(sums)
    assert abs(signal[0] / power - 1) < 0.05
    assert abs(kappa_hat[0] / kappa - 1) < 0.03
    assert not clamped[0]


def test_the_estimates_do_not_depend_on_the_batch_size():
    """Half batches of 0.5, 2 and 8 effective particles per voxel (b = 300, 1000, 3000 roughly scale F_h)."""
    estimates = []
    for scale in (0.5, 2.0, 8.0):
        rng = np.random.default_rng(11)
        sums = _simulate(rng, n_voxels=200_000, power=0.4, kappa=0.7, particles_per_half=scale, batches=6)
        signal, kappa_hat, _ = signal_and_noise(sums)
        estimates.append((signal[0], kappa_hat[0]))
    estimates = np.asarray(estimates)
    assert np.allclose(estimates[:, 0], 0.4, rtol=0.06)
    assert np.allclose(estimates[:, 1], 0.7, rtol=0.04)


def test_a_pure_noise_shell_gives_zero_signal_up_to_sampling_and_the_clamp_reports_itself():
    rng = np.random.default_rng(3)
    sums = _simulate(rng, n_voxels=100_000, power=0.0, kappa=1.0, particles_per_half=2.0, batches=4, model_scale=0.0)
    raw, kappa_hat, _ = signal_and_noise(sums)
    assert abs(raw[0]) < 0.01
    signal, _, clamped = signal_and_noise(sums, floor=1e-3)
    assert signal[0] >= 1e-3 and clamped[0] == (raw[0] < 1e-3)
    assert abs(kappa_hat[0] - 1) < 0.03


def test_the_signal_estimate_does_not_see_the_model():
    """The shared model cancels: the same data with two different models give the same sums."""
    rng = np.random.default_rng(5)
    n = 1000
    weight = rng.gamma(2.0, 1.0, (2, n))
    rhs = _complex(rng, 1.0, (2, n)) * weight
    shells = np.zeros(n, np.int64)
    a = shell_sums(rhs - weight * 0.0, weight, np.zeros(n), 0.5, shells, 1)
    model = _complex(rng, 3.0, n)
    b = shell_sums(rhs - weight * model[None], weight, model, 0.5, shells, 1)
    assert np.allclose(a, b)
