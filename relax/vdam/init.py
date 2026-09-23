"""Denovo InitialModel state initialisation (RELION parity).

Mirrors MlModel::initialiseFromImages fn_ref=None branch
(ml_model.cpp:1082) plus the surrounding ini_high / sigma2_noise / tau2
setup. Three public callables:

- ``initialise_denovo_state`` — particle-independent state fields.
- ``seed_noise_from_mavg`` — write per-optics-group sigma2_noise.
- ``initialise_data_vs_prior_from_references`` — seed tau2_class (shared per class with auto-refine).
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from relax.vdam.state import MOM2_INIT_CONSTANT, InitialModelState, half_slot_count

# RELION's 0.07 digital-frequency low-pass for do_average_unaligned (ml_optimiser.cpp:2513-2518).
INI_HIGH_DIGITAL_FREQ: float = 0.07


def compute_ini_high_shell(ori_size: int) -> int:
    """``ROUND(0.07 * ori_size)`` (initial resolution shell)."""
    return int(math.floor(INI_HIGH_DIGITAL_FREQ * ori_size + 0.5))


def compute_ini_high_angstrom(ori_size: int, pixel_size: float) -> float:
    """Initial low-pass in Ångström (1/getResolution(ini_shell))."""
    return ori_size * pixel_size / compute_ini_high_shell(ori_size)


def compute_current_size_for_denovo(ori_size: int) -> int:
    """Pre-iter-1 ``current_size = 2*(ini_shell + 10)`` clipped to ``ori_size``."""
    return min(2 * (compute_ini_high_shell(ori_size) + 10), ori_size)


def initialise_denovo_state(
    *,
    ori_size: int,
    pixel_size: float,
    K: int,
    nr_iter: int,
    n_directions: int,
    nr_optics_groups: int = 1,
    pseudo_halfsets: bool = True,
    padding_factor: int = 1,
) -> InitialModelState:
    """Fresh ``InitialModelState`` at iter 0 (ml_model.cpp:1082-1133 denovo branch)."""
    if K < 1:
        raise ValueError("K must be >= 1")
    if n_directions < 1:
        raise ValueError("n_directions must be >= 1")
    if ori_size < 2:
        raise ValueError("ori_size must be >= 2")

    pf = padding_factor
    pad_shape = (ori_size * pf, ori_size * pf, (ori_size * pf) // 2 + 1)
    n_shells = ori_size // 2 + 1
    ini_high_A = compute_ini_high_angstrom(ori_size, pixel_size)

    return InitialModelState(
        iter=0,
        nr_iter=nr_iter,
        K=K,
        ori_size=ori_size,
        pixel_size=pixel_size,
        pseudo_halfsets=pseudo_halfsets,
        Iref=np.zeros((K, ori_size, ori_size, ori_size), dtype=np.float64),
        Igrad1=np.zeros((half_slot_count(K, pseudo_halfsets), *pad_shape), dtype=np.complex128),
        Igrad2=np.full((K, *pad_shape), MOM2_INIT_CONSTANT + 1j * MOM2_INIT_CONSTANT, dtype=np.complex128),
        sigma2_noise=np.zeros((nr_optics_groups, n_shells), dtype=np.float64),
        tau2_class=np.zeros((K, n_shells), dtype=np.float64),
        sigma2_class=np.zeros((K, n_shells), dtype=np.float64),
        fsc_halves_class=np.zeros((K, n_shells), dtype=np.float64),
        fourier_coverage_class=np.zeros((K, n_shells), dtype=np.float64),
        data_vs_prior_class=np.zeros((K, n_shells), dtype=np.float64),
        pdf_class=np.full(K, 1.0 / K, dtype=np.float64),
        pdf_direction=np.full((K, n_directions), 1.0 / (K * n_directions), dtype=np.float64),
        sigma2_offset=100.0,
        ini_high=ini_high_A,
        current_resolution=1.0 / ini_high_A,
        current_resolution_shell=compute_ini_high_shell(ori_size),
        current_size=compute_current_size_for_denovo(ori_size),
    )


def seed_noise_from_mavg(
    state: InitialModelState,
    sigma2_per_group: np.ndarray,
) -> InitialModelState:
    """Write ``sigma2_noise`` from an externally-computed ``(G, S)`` spectrum."""
    if sigma2_per_group.shape != state.sigma2_noise.shape:
        raise ValueError(f"sigma2_per_group shape {sigma2_per_group.shape} != expected {state.sigma2_noise.shape}")
    new_state = replace(state)
    new_state.sigma2_noise = np.asarray(sigma2_per_group, dtype=np.float64).copy()
    return new_state


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
    radius = np.sqrt(kz[:, None, None] ** 2 + kz[None, :, None] ** 2 + kx[None, None, :] ** 2)
    shell = np.floor(radius + 0.5).astype(np.int64)
    valid = shell < int(n_shells)
    out = np.zeros(int(n_shells), dtype=np.float64)
    count = np.zeros(int(n_shells), dtype=np.float64)
    power = np.abs(fourier) ** 2
    np.add.at(out, shell[valid].ravel(), power[valid].ravel())
    np.add.at(count, shell[valid].ravel(), 1.0)
    nz = count > 0.0
    out[nz] /= count[nz]
    return out


def relion_initial_tau2_and_data_vs_prior(
    iref_relion: np.ndarray,
    *,
    tau2_fudge: float,
    avg_sigma2_noise: np.ndarray,
    nr_particles: int,
    pdf_class: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """One class of ``MlModel::initialiseDataVersusPrior`` (ml_model.cpp:1557-1626).

    ``iref_relion`` is the start-up reference in RELION's frame (after
    ``initialLowPassFilterReferences``); ``avg_sigma2_noise`` is the RELION-unit
    noise spectrum averaged over optics groups with noise. Returns the RELION-unit
    ``tau2_class`` and ``data_vs_prior_class`` spectra, ``ori_size // 2 + 1`` shells.
    """
    iref = np.asarray(iref_relion, dtype=np.float64)
    ori_size = int(iref.shape[0])
    avg_sigma2_noise = np.asarray(avg_sigma2_noise, dtype=np.float64).reshape(-1)
    if avg_sigma2_noise.shape != (ori_size // 2 + 1,) or np.any(avg_sigma2_noise <= 0.0) or nr_particles <= 0:
        raise ValueError("need a positive noise spectrum of ori_size // 2 + 1 shells and nr_particles > 0")
    spectrum = _relion_power_spectrum_3d(iref, ori_size // 2 + 1)
    spectrum *= float(ori_size * ori_size) / 2.0
    tau2 = float(tau2_fudge) * spectrum
    return tau2, _relion_data_vs_prior(tau2, avg_sigma2_noise, nr_particles, pdf_class)


def _relion_data_vs_prior(tau2, avg_sigma2_noise, nr_particles, pdf_class):
    """``data_vs_prior = N * pdf_class * tau2 / (sigma2 * 2i)``, shell 0 without the ``2i``."""
    shell_factor = np.maximum(1.0, 2.0 * np.arange(avg_sigma2_noise.size, dtype=np.float64))
    evidence = float(nr_particles) * float(pdf_class) / avg_sigma2_noise
    return evidence / shell_factor * tau2


def initialise_data_vs_prior_from_references(
    state: InitialModelState,
    *,
    nr_particles: int,
    fix_tau: bool = False,
) -> InitialModelState:
    """``MlModel::initialiseDataVersusPrior`` (avoids the 0.001*weight fallback if tau2=0)."""
    if nr_particles <= 0:
        raise ValueError(f"nr_particles must be positive, got {nr_particles}")
    sigma2 = np.asarray(state.sigma2_noise, dtype=np.float64)
    if sigma2.ndim != 2 or sigma2.shape[1] != state.ori_size // 2 + 1:
        raise ValueError(f"sigma2_noise must have shape (G, {state.ori_size // 2 + 1}), got {sigma2.shape}")
    group_has_noise = np.sum(sigma2, axis=1) > 0.0
    if not np.any(group_has_noise):
        raise ValueError("cannot initialise data_vs_prior without positive sigma2_noise")
    avg_sigma2_noise = np.mean(sigma2[group_has_noise], axis=0)
    if np.any(avg_sigma2_noise <= 0.0):
        raise ValueError("avg sigma2_noise must be positive in all Fourier shells")

    new_tau2 = np.asarray(state.tau2_class, dtype=np.float64).copy()
    new_data_vs_prior = np.zeros_like(new_tau2)
    pdf_class = np.asarray(state.pdf_class, dtype=np.float64)
    if pdf_class.shape != (state.K,):
        raise ValueError(f"pdf_class must have shape ({state.K},), got {pdf_class.shape}")

    from recovar.utils.helpers import recovar_volume_to_relion

    for k in range(int(state.K)):
        if not fix_tau:
            # getSpectrum runs on RELION-frame Iref.  Its half-spectrum shell
            # traversal is not invariant to swapping RECOVAR's X/Z axes, so
            # convert at this native boundary just as reconstruction does.
            iref_relion = recovar_volume_to_relion(np.asarray(state.Iref[k], dtype=np.float64))
            new_tau2[k], new_data_vs_prior[k] = relion_initial_tau2_and_data_vs_prior(
                iref_relion,
                tau2_fudge=float(state.tau2_fudge_factor),
                avg_sigma2_noise=avg_sigma2_noise,
                nr_particles=nr_particles,
                pdf_class=float(pdf_class[k]),
            )
        else:
            new_data_vs_prior[k] = _relion_data_vs_prior(new_tau2[k], avg_sigma2_noise, nr_particles, pdf_class[k])
        if np.any(new_tau2[k] < 0.0):
            raise ValueError("initial tau2_class must be non-negative after reference-spectrum initialisation")

    new_state = replace(state)
    new_state.tau2_class = new_tau2
    new_state.data_vs_prior_class = new_data_vs_prior
    return new_state
