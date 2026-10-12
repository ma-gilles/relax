"""Fixed grid and taper conventions used by RELION refinement.

Image masking uses image pixels; initial reference filtering uses Fourier
shells. The values follow ``WIDTH_MASK_EDGE`` and ``WIDTH_FMASK_EDGE`` in
RELION's ``ml_optimiser.h``. Projection zero-pads the real-space reference;
reconstruction pads the Fourier backprojection. Both use ``--pad 2``.
The soft image and map mask is :func:`soft_mask_weights`.
"""

import numpy as np

from relax.relion.macros import PI

IMAGE_MASK_EDGE_PIXELS = 5
REFERENCE_FILTER_EDGE_SHELLS = 2
PROJECTION_PADDING_FACTOR = 2
RECONSTRUCTION_PADDING_FACTOR = 2


def soft_mask_weights(shape, radius: float, width: float):
    """RELION's ``softMaskOutsideMap`` background weight at every voxel of a ``shape`` grid (2D or 3D), float64.

    ``w = 0`` for ``r < radius``, ``1`` for ``r > radius + width`` and ``0.5 + 0.5 cos(PI (radius + width - r) / width)``
    between (mask.cpp ``softMaskOutsideMap``); a masked map keeps ``1 - w`` of its value and takes ``w`` of the
    background. ``r`` is the distance from the Xmipp origin (logical index ``-int(n / 2)`` first on each axis), the
    square root of the integer squared distance. A zero ``width`` gives NaN at ``r == radius``, RELION's 0/0. The
    caller resolves RELION's default radius (``radius < 0``: half the box).
    """
    axes = [np.arange(int(n)) - int(float(n) / 2.0) for n in shape]
    grids = np.meshgrid(*axes, indexing="ij")
    r = np.sqrt(sum(g.astype(np.int64) * g for g in grids).astype(np.float64))
    radius_p = radius + width
    outside = r > radius_p
    edge = ~outside & ~(r < radius)
    with np.errstate(divide="ignore", invalid="ignore"):
        raised = np.where(edge, 0.5 + 0.5 * np.cos(PI * (radius_p - r) / width), 0.0)
    return np.where(outside, 1.0, raised)
