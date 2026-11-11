"""shift program: truth-free model centring helpers (RELION centerClasses analogue) on relax's half-Fourier theta columns."""
import numpy as np
from recovar.core import fourier_transform_utils as ftu


def half_kgrid(volume_shape):
    """(n_half, 3) integer frequency coords of the packed half volume, same axis order as the real array."""
    N = volume_shape[0]
    full = np.stack(np.meshgrid(*[np.arange(n) - n // 2 for n in volume_shape], indexing='ij'), -1).reshape(-1, 3).astype(np.float64)
    out = []
    for ax in range(3):
        out.append(np.asarray(ftu.full_volume_to_half_volume(full[:, ax].reshape(volume_shape).astype(np.complex64), volume_shape)).real)
    return np.stack([o.reshape(-1) for o in out], -1), N


def real_mean(theta_mean_half, volume_shape):
    full = np.asarray(ftu.half_volume_to_full_volume(theta_mean_half, volume_shape)).reshape(volume_shape)
    return np.real(np.asarray(ftu.get_idft3(full)))


def com_px(vol, positive=True):
    v = np.clip(vol, 0, None) if positive else vol
    idx = np.stack(np.meshgrid(*[np.arange(n) - n // 2 for n in vol.shape], indexing='ij'), 0)
    return np.array([np.sum(v * idx[a]) for a in range(3)]) / max(np.sum(v), 1e-30)


def shift_half(theta_half_cols, kgrid, N, d):
    """shift real-space content by +d voxels (array axis order): multiply by exp(-2 pi i k.d / N)."""
    ph = np.exp(-2j * np.pi * (kgrid @ np.asarray(d, np.float64)) / N)
    return theta_half_cols * ph[:, None].astype(np.complex64)
