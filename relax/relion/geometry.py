"""Fixed grid and taper conventions used by RELION refinement.

Image masking uses image pixels; initial reference filtering uses Fourier
shells. The values follow ``WIDTH_MASK_EDGE`` and ``WIDTH_FMASK_EDGE`` in
RELION's ``ml_optimiser.h``. Projection zero-pads the real-space reference;
reconstruction pads the Fourier backprojection. Both use ``--pad 2``.
"""

IMAGE_MASK_EDGE_PIXELS = 5
REFERENCE_FILTER_EDGE_SHELLS = 2
PROJECTION_PADDING_FACTOR = 2
RECONSTRUCTION_PADDING_FACTOR = 2
