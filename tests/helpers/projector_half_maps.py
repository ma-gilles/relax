"""References converted to RELION projector half maps without their power spectrum, for tests.

Moved from relax/relion/projector_setup.py, where no relax module used it: relax calls
``reference_to_relion_projector_half_maps_and_power``, which this wraps.
"""

import numpy as np

from relax.relion.projector_setup import reference_to_relion_projector_half_maps_and_power
from relax.runtime.engine_tuning import DEFAULT_STABLE_FOURIER_WINDOW_QUANTUM


def reference_to_relion_projector_half_maps(
    references: np.ndarray,
    *,
    current_size: int,
    padding_factor: int = 1,
    interpolator: int = 1,
    projector_data_dtype=None,
    compute_dtype=np.float64,
) -> tuple[np.ndarray, int]:
    """Convert references to RELION half maps without retaining their spectrum."""
    half_maps, _power, r_max = reference_to_relion_projector_half_maps_and_power(
        references,
        current_size=current_size,
        padding_factor=padding_factor,
        interpolator=interpolator,
        projector_data_dtype=projector_data_dtype,
        compute_dtype=compute_dtype,
        stable_window_quantum=DEFAULT_STABLE_FOURIER_WINDOW_QUANTUM,
    )
    return half_maps, r_max
