"""Noise in unnormalized full-real-image Fourier units.

See docs/math/vdam_ppca_algorithm.md section 10. A real image with independent
pixel variance v has E|FFT(epsilon)[k]|^2 = H*W*v, including self-conjugate
frequencies. Hermitian multiplicities count real degrees of freedom; they do
not turn this total coefficient variance into a per-component variance.
"""

import jax.numpy as jnp
import numpy as np


def relion_to_coefficient_variance(sigma2, image_shape):
    """Convert normalized RELION per-component variance to E|FFT(epsilon)|²."""
    return jnp.asarray(sigma2, dtype=jnp.float32) * np.float32(2 * np.prod(image_shape) ** 2)


def expected_residual_power(y, projected_mean, projected_loadings, mean, covariance):
    """Expected squared residual, retaining posterior coordinate uncertainty.

    Pixel dimension is last in y/mean projection and penultimate in loadings;
    leading dimensions broadcast. No division by CTF (which may have zeros).
    """
    residual = y - projected_mean - jnp.einsum("...fq,...q->...f", projected_loadings, mean)
    uncertainty = jnp.einsum("...fq,...qp,...fp->...f", projected_loadings, covariance, projected_loadings.conj()).real
    return jnp.abs(residual) ** 2 + uncertainty


def update_noise(previous, numerator, denominator, *, full_data=False):
    """Apply VDAM timing to raw shell sums at the old E-step model."""
    numerator, denominator = np.asarray(numerator), np.asarray(denominator)
    if np.any(denominator <= 0):
        raise ValueError("Noise shells must have positive measured counts")
    measured = numerator / denominator
    if not np.all(np.isfinite(measured)) or np.any(measured <= 0):
        raise ValueError("Nonpositive/nonfinite expected noise variance")
    beta = np.float32(0 if full_data else 0.9)
    return jnp.asarray(beta * np.asarray(previous) + (1 - beta) * measured, dtype=jnp.float32)
