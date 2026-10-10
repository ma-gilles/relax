"""Reference filtering and starting spectra shared by refinement and VDAM."""

from __future__ import annotations

import numpy as np

from relax.fourier.shells import shell_of_radius_sq
from relax.relion.geometry import REFERENCE_FILTER_EDGE_SHELLS

# Elements of the start-up low-pass mask built at once (128 MiB of float64).
_MASK_SLAB_ELEMENTS = 1 << 24


def initial_low_pass_filter_references(
    Iref: np.ndarray,
    *,
    box_size: int,
    pixel_size: float,
    ini_high_ang: float,
    filter_edgewidth: float = REFERENCE_FILTER_EDGE_SHELLS,
) -> np.ndarray:
    """``initialLowPassFilterReferences`` (ml_optimiser.cpp:3336): cosine-taper from r=radius outward to r=radius_p."""
    edge_width = float(filter_edgewidth)
    radius = box_size * pixel_size / ini_high_ang - edge_width / 2.0
    radius_p = radius + edge_width
    N = Iref.shape[1]
    kz = np.fft.fftfreq(N, d=1.0) * N
    kx = np.arange(N // 2 + 1, dtype=np.float64)

    def mask_rows(z0, z1):
        r = np.sqrt(kz[z0:z1, None, None] ** 2 + kz[None, :, None] ** 2 + kx[None, None, :] ** 2)
        mask = np.zeros_like(r)
        mask[r < radius] = 1.0
        edge = (r >= radius) & (r <= radius_p)
        if edge_width > 0:
            mask[edge] = 0.5 - 0.5 * np.cos(np.pi * (radius_p - r[edge]) / edge_width)
        return mask

    # The products run in place, in the order of ``F / size * mask * size`` and with the mask built a slab of
    # rows at a time, so each value is rounded as before; the inverse transform writes into ``out`` (relax#39:
    # the out-of-place temporaries of a box-800 start-up).
    rows = max(1, _MASK_SLAB_ELEMENTS // (N * (N // 2 + 1)))
    out = np.zeros_like(Iref)
    for k in range(Iref.shape[0]):
        vol = Iref[k]
        F = np.fft.rfftn(vol, axes=(0, 1, 2), norm=None)
        F /= vol.size
        for z0 in range(0, N, rows):
            F[z0 : z0 + rows] *= mask_rows(z0, min(N, z0 + rows))
        F *= vol.size
        np.fft.irfftn(F, s=vol.shape, axes=(0, 1, 2), norm=None, out=out[k])
        del F
    return out



def _relion_power_spectrum_3d(volume: np.ndarray, n_shells: int) -> np.ndarray:
    """RELION ``getSpectrum(POWER_SPECTRUM)``: FFTW-normalized forward FFT, per-shell mean ``|F|^2``."""
    vol = np.asarray(volume, dtype=np.float64)
    if vol.ndim != 3 or vol.shape[0] != vol.shape[1] or vol.shape[1] != vol.shape[2]:
        raise ValueError(f"volume must be cubic 3D, got shape {vol.shape}")
    n = int(vol.shape[0])
    if n_shells < 1:
        raise ValueError(f"n_shells must be positive, got {n_shells}")

    fourier = np.fft.rfftn(vol, axes=(0, 1, 2), norm=None) / float(vol.size)
    kz = np.fft.fftfreq(n, d=1.0) * n
    kx = np.arange(n // 2 + 1, dtype=np.float64)
    shell = shell_of_radius_sq(kz[:, None, None] ** 2 + kz[None, :, None] ** 2 + kx[None, None, :] ** 2, rule="half_up")
    valid = shell < int(n_shells)
    out = np.zeros(int(n_shells), dtype=np.float64)
    count = np.zeros(int(n_shells), dtype=np.float64)
    power = np.abs(fourier) ** 2
    np.add.at(out, shell[valid].ravel(), power[valid].ravel())
    np.add.at(count, shell[valid].ravel(), 1.0)
    nz = count > 0.0
    out[nz] /= count[nz]
    return out



def _whole_grid_power_spectrum_3d(volume: np.ndarray, n_shells: int) -> np.ndarray:
    """:func:`_relion_power_spectrum_3d` with every Hermitian pair counted once.

    ``getSpectrum`` (fftw.cpp:1010-1039) visits every coefficient of the FFTW half once, so the pairs
    whose two members are both stored (the ``kx = 0`` and Nyquist ``kx`` planes) count twice. This is
    the shell mean of ``|F|^2`` over the whole Fourier grid, every coefficient once.
    """
    vol = np.asarray(volume, dtype=np.float64)
    n = int(vol.shape[0])
    power = np.abs(np.fft.fftn(vol) / float(vol.size)) ** 2
    k = np.fft.fftfreq(n, d=1.0) * n
    shell = shell_of_radius_sq(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2, rule="half_up")
    valid = shell < int(n_shells)
    total = np.bincount(shell[valid], weights=power[valid], minlength=int(n_shells))
    count = np.bincount(shell[valid], minlength=int(n_shells))
    return np.where(count > 0, total / np.maximum(count, 1), 0.0)


_POWER_SPECTRUM_3D = {"relion": _relion_power_spectrum_3d, "once": _whole_grid_power_spectrum_3d}


def relion_initial_tau2_and_data_vs_prior(
    iref_relion: np.ndarray,
    *,
    tau2_fudge: float,
    avg_sigma2_noise: np.ndarray,
    nr_particles: int,
    pdf_class: float = 1.0,
    shell_pair_counting: str = "relion",
) -> tuple[np.ndarray, np.ndarray]:
    """One class of ``MlModel::initialiseDataVersusPrior`` (ml_model.cpp:1557-1626).
    The reference uses RELION's frame after low-pass filtering; noise averages optics groups with noise.
    ``shell_pair_counting`` is the pair counting of its power spectrum ("relion" or "once").
    Return RELION-unit tau2 and data/prior spectra, each with ``ori_size // 2 + 1`` shells.
    """
    iref = np.asarray(iref_relion, dtype=np.float64)
    box_size = int(iref.shape[0])
    avg_sigma2_noise = np.asarray(avg_sigma2_noise, dtype=np.float64).reshape(-1)
    if avg_sigma2_noise.shape != (box_size // 2 + 1,) or np.any(avg_sigma2_noise <= 0.0) or nr_particles <= 0:
        raise ValueError("need a positive noise spectrum of ori_size // 2 + 1 shells and nr_particles > 0")
    spectrum = _POWER_SPECTRUM_3D[shell_pair_counting](iref, box_size // 2 + 1)
    spectrum *= float(box_size * box_size) / 2.0
    tau2 = float(tau2_fudge) * spectrum
    return tau2, _relion_data_vs_prior(tau2, avg_sigma2_noise, nr_particles, pdf_class)



def _relion_data_vs_prior(tau2, avg_sigma2_noise, nr_particles, pdf_class):
    """``data_vs_prior = N * pdf_class * tau2 / (sigma2 * 2i)``, shell 0 without the ``2i``."""
    shell_factor = np.maximum(1.0, 2.0 * np.arange(avg_sigma2_noise.size, dtype=np.float64))
    evidence = float(nr_particles) * float(pdf_class) / avg_sigma2_noise
    return evidence / shell_factor * tau2
