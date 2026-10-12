"""Half-spectrum translation phases for dense EM engines (RELION's lattice labels, the phase table and its
application).

The batch preprocessing that reads the dataset's optics, and the image-backend selectors, are
:mod:`relax.relion.image_preprocessing`.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import recovar.core.fourier_transform_utils as fourier_transform_utils

from relax.cuda import (
    kernels as _em_cuda_kernels,  # noqa: F401  (registers the relion_cuda preprocessor, relax split seam S2)
)


@jax.jit
def apply_half_translation_phases(weighted_half, translation_phases_half):
    return (weighted_half[:, None, :] * translation_phases_half[None, :, :]).reshape(
        weighted_half.shape[0] * translation_phases_half.shape[0],
        weighted_half.shape[1],
    )


def relion_half_translation_lattice(image_shape):
    """Packed half-spectrum frequencies with RELION's row and column labels.

    ``get_k_coordinate_of_each_pixel_half`` labels the packed Nyquist row
    ``ky = -N/2`` and the packed Nyquist column ``kx = -N/2``, which is RECOVAR's
    own centered convention and is what the non-EM callers of that core helper
    expect. RELION labels the same physical row of a half image it is not
    cropping ``+N/2`` (``fftw.h:99-109``: ``ip = (i < XSIZE) ? i : i - YSIZE``
    with ``XSIZE`` the half width) and its column ``x = j`` up to ``+N/2``, and
    its scoring and translate kernels all derive those labels, so an EM
    translation phase built on the centered labels is the conjugate of RELION's
    on that row and column for any shift that is not a whole pixel.

    The table is always built on the uncropped packed half, so RELION's labels
    are the right ones for it; a cropped window simply never selects that row
    or column (``windowFourierTransform`` keeps ``ip = -(cs/2-1)..+cs/2``,
    ``fftw.h:849-855``). This wrapper is EM-local by design: the core helper
    keeps its own convention for its other callers.
    """

    lattice = jnp.asarray(
        fourier_transform_utils.get_k_coordinate_of_each_pixel_half(
            image_shape, voxel_size=1, scaled=True
        )
    )
    box_size = int(image_shape[0])
    if box_size % 2 == 0:
        for axis in (1, 0):
            k = jnp.rint(lattice[:, axis] * box_size).astype(jnp.int32)
            nyquist = k == -(box_size // 2)
            lattice = lattice.at[:, axis].set(jnp.where(nyquist, -lattice[:, axis], lattice[:, axis]))
    return lattice


def half_translation_phase_table(translations, image_shape, dtype=jnp.float32):
    """Return the complex phase-shift table for a translation grid.

    ``dtype`` defaults to float32 (RELION's accelerated-GPU precision, and
    RECOVAR's own historical default here). Pass ``jnp.float64`` to compute
    genuinely double-precision phase factors; ``jax.lax.Precision.HIGHEST``
    alone does not upcast float32 inputs, so the input dtype must change too.

    Frequencies come from :func:`relion_half_translation_lattice`, so the
    packed Nyquist row carries RELION's label rather than RECOVAR's centered
    one.
    """
    real_dtype = jnp.float64 if jnp.dtype(dtype) == jnp.dtype(jnp.float64) else jnp.float32
    return _half_translation_phase_table(
        jnp.asarray(translations), tuple(int(v) for v in image_shape), np.dtype(real_dtype).name
    )


@partial(jax.jit, static_argnames=("image_shape", "real_dtype"))
def _half_translation_phase_table(translations, image_shape, real_dtype):
    # One program per (translations, image shape): eager, the lattice and the
    # phases were about twenty single-primitive programs per new image size.
    real_dtype = jnp.dtype(real_dtype)
    complex_dtype = jnp.complex128 if real_dtype == jnp.float64 else jnp.complex64
    phase_arg = jnp.einsum(
        "td,pd->tp",
        translations.astype(real_dtype),
        relion_half_translation_lattice(image_shape).astype(real_dtype),
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.exp(jnp.asarray(-2j * jnp.pi, dtype=complex_dtype) * phase_arg)
