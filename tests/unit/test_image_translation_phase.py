"""The host translation phase (:func:`relax.helpers.preprocessing.half_translation_phase_table`) is RELION's.

The reference here is written from RELION's kernels, not from relax: ``translatePixel`` multiplies
pixel ``(x, y)`` by ``exp(i v)``, ``v = x * tx + y * ty`` with ``tx = -2 pi xshift / N``
(acc_ml_optimiser_impl.h:1549-1550, acc/cuda/cuda_kernels/cuda_device_utils.cuh), ``x`` the
half-width column and ``y`` the row labelled ``i`` up to ``N/2`` and ``i - N`` above
(fftw.h:99-109), on NumPy's unshifted ``rfft2`` layout. A whole-pixel translation must also be
the circular shift of the image itself.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit

# The lattice holds k / N in float32 (get_k_coordinate_of_each_pixel_half), so the phase carries a float32
# rounding of the angle: below 5e-7 absolute at N = 9 (coefficients up to about 20).
_ATOL = 2e-6


def _relion_translated_rfft(image, shift):
    """RELION's translation sample of ``image`` by ``shift`` (x, y pixels), in NumPy's rfft2 layout."""
    n = image.shape[0]
    spectrum = np.fft.rfft2(np.fft.fftshift(image))
    rows = np.arange(n)
    y = np.where(rows <= n // 2, rows, rows - n)[:, None]
    x = np.arange(n // 2 + 1)[None, :]
    v = x * (-2 * np.pi * shift[0] / n) + y * (-2 * np.pi * shift[1] / n)
    return spectrum * np.exp(1j * v)


def _relax_translated_rfft(image, shift):
    """relax's packed half of ``image`` times the image-translation phase, back in NumPy's rfft2 layout."""
    import jax.numpy as jnp
    from recovar.core import fourier_transform_utils as ftu

    from relax.helpers.preprocessing import half_translation_phase_table

    n = image.shape[0]
    half = np.asarray(ftu.get_dft2_real(jnp.asarray(image, dtype=jnp.float64))).reshape(-1)
    phase = np.asarray(half_translation_phase_table(np.asarray([shift], np.float64), (n, n), dtype=jnp.float64))[0]
    # get_dft2_real centres the rows of NumPy's rfft2 (fftshift on axis -2) and nothing else.
    return np.fft.ifftshift((half * phase).reshape(n, n // 2 + 1), axes=0)


@pytest.mark.parametrize("n", [8, 9, 16])
@pytest.mark.parametrize("shift", [(0.3, -0.45), (-0.5, 0.5), (0.17, 0.0), (0.0, -0.21)])
def test_image_translation_phase_is_relions_translation_sample(n, shift):
    image = np.random.default_rng(n).normal(size=(n, n))
    actual = _relax_translated_rfft(image, shift)
    desired = _relion_translated_rfft(image, shift)
    # Every packed pixel, the Nyquist row and column included.
    np.testing.assert_allclose(actual, desired, rtol=0, atol=_ATOL * np.abs(desired).max())


@pytest.mark.parametrize("shift", [(1, 0), (0, -2), (3, 1)])
def test_a_whole_pixel_translation_is_the_circular_shift(shift):
    n = 8
    image = np.random.default_rng(1).normal(size=(n, n))
    # Image x is the column axis, y the row axis; the sample moves the content by +shift.
    rolled = np.roll(image, (shift[1], shift[0]), axis=(0, 1))
    desired = np.fft.rfft2(np.fft.fftshift(rolled))
    np.testing.assert_allclose(
        _relax_translated_rfft(image, shift), desired, rtol=0, atol=_ATOL * np.abs(desired).max()
    )
    np.testing.assert_allclose(
        _relion_translated_rfft(image, shift), desired, rtol=0, atol=_ATOL * np.abs(desired).max()
    )
