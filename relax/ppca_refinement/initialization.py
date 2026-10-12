"""Half-Fourier storage conversion of one real-space volume, shared by the PPCA dense iteration."""

from __future__ import annotations

import numpy as np
import recovar.core.fourier_transform_utils as ftu


def real_volume_to_centered_fourier(volume: np.ndarray) -> np.ndarray:
    """Convert a recovar-frame real-space volume to centered Fourier storage."""
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3D, got {volume.shape}")
    return np.asarray(ftu.get_dft3(volume), dtype=np.complex64)


def real_volume_to_centered_fourier_half(volume: np.ndarray) -> np.ndarray:
    """Convert a real-space volume to flattened RECOVAR half-Fourier storage."""
    centered = real_volume_to_centered_fourier(volume)
    return np.asarray(ftu.full_volume_to_half_volume(centered, centered.shape), dtype=np.complex64).reshape(-1)
