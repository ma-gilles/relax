"""RELION's own ``Projector::computeFourierTransformMap`` through the binding, as an oracle.

Production builds the projector with
:func:`relax.relion.relion_projector_setup.reference_to_relion_projector_half_maps_and_power`;
tests and parity tools compare it with this.
"""

from __future__ import annotations

import numpy as np


def native_reference_to_relion_projector_half_maps_and_power(
    references,
    *,
    current_size: int,
    padding_factor: int = 1,
    interpolator: int = 1,
    projector_data_dtype=np.complex128,
) -> tuple[np.ndarray, np.ndarray, int]:
    """RELION-frame ``(half_maps, power, r_max)`` of RECOVAR-frame references, from RELION's code."""

    from recovar.utils.helpers import recovar_volume_to_relion

    from relax.relion_bind import _relion_bind_core as bind

    halves, power_spectra, r_max_values = [], [], []
    for ref in np.asarray(references):
        projector_data, power, _ori, _pad, r_max, _r_min_nn, _interp = bind.compute_fourier_transform_map(
            np.asarray(recovar_volume_to_relion(ref), dtype=np.float64),
            int(ref.shape[-1]),
            int(padding_factor),
            int(interpolator),
            int(current_size),
            True,
            2,
        )
        halves.append(np.asarray(projector_data).astype(np.dtype(projector_data_dtype), copy=False))
        power_spectra.append(np.asarray(power, dtype=np.float64))
        r_max_values.append(int(r_max))
    if len(set(r_max_values)) != 1:
        raise ValueError(f"RELION projector maps disagree on r_max: {r_max_values}")
    return np.asarray(halves), np.asarray(power_spectra, dtype=np.float64), int(r_max_values[0])


def native_reference_to_relion_projector_half_maps(references, **kwargs) -> tuple[np.ndarray, int]:
    """:func:`native_reference_to_relion_projector_half_maps_and_power` without the spectrum."""

    half_maps, _power, r_max = native_reference_to_relion_projector_half_maps_and_power(references, **kwargs)
    return half_maps, r_max
