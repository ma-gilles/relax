"""RELION initial noise estimation and process-start noise inputs (pure NumPy).

Mirrors RELION ``calculateSumOfPowerSpectraAndAverageImage``
(ml_optimiser.cpp:2891) + ``setSigmaNoiseEstimatesAndSetAverageImage``
(ml_optimiser.cpp:3243). For up to N≤1000 particles per optics group:
accumulate Mavg and per-shell radial power; then
``sigma2_noise[g] = sum_sigma2[g] / (2 * sumw[g]) - 0.5*|FFT(Mavg)|²``
with negative shells replaced by their nearest positive neighbour.

Compute initial image statistics or read the supported single-optics noise
spectrum; preserve RELION MPI half-set initialization semantics.
"""

from __future__ import annotations

import re
from typing import Iterable, Iterator, Tuple

import numpy as np


def _softmask_outside_map(image: np.ndarray, radius: float, cosine_width: float) -> np.ndarray:
    """RELION ``softMaskOutsideMap`` (mask.cpp:43): cosine taper between ``r=radius`` and ``radius+cosine_width``."""
    H, W = image.shape[-2:]
    yy = np.arange(H) - H // 2
    xx = np.arange(W) - W // 2
    r = np.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2).astype(np.float64)
    radius_p = radius + cosine_width

    w = np.zeros_like(r)
    outside = r > radius_p
    edge = (r >= radius) & (r <= radius_p)
    w[outside] = 1.0
    if cosine_width > 0 and edge.any():
        w[edge] = 0.5 + 0.5 * np.cos(np.pi * (radius_p - r[edge]) / cosine_width)

    w_sum = w.sum()
    avg_bg = float((w * image).sum() / w_sum) if w_sum > 0 else 0.0

    out = image.astype(np.float64, copy=True)
    out[outside] = avg_bg
    if cosine_width > 0 and edge.any():
        out[edge] = (1.0 - w[edge]) * image[edge] + w[edge] * avg_bg
    return out.astype(image.dtype)


def _radial_power_spectrum(image_real: np.ndarray, n_shells: int) -> np.ndarray:
    """Per-shell mean ``|FFT(image)|²`` (ml_optimiser.cpp:3108-3117); RELION-normalised by ``H*W``."""
    H, W = image_real.shape[-2:]
    F = np.fft.rfft2(image_real, norm=None) / (H * W)
    ky = np.fft.fftfreq(H, d=1.0) * H
    kx = np.arange(W // 2 + 1, dtype=np.float64)
    ires = np.round(np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)).astype(np.int64)
    out = np.zeros(n_shells, dtype=np.float64)
    count = np.zeros(n_shells, dtype=np.int64)
    np.add.at(out, ires[ires < n_shells], (F.real**2 + F.imag**2)[ires < n_shells])
    flat = ires.ravel()
    np.add.at(count, flat[flat < n_shells], 1)
    count[count == 0] = 1
    return out / count


def _fix_negative_sigma2(sigma2: np.ndarray) -> np.ndarray:
    """Replace non-positive shells with the nearest positive neighbour (ml_optimiser.cpp:3293-3320)."""
    out = sigma2.copy()
    n = out.size
    for i in range(n):
        if out[i] > 0.0:
            continue
        if i - 1 >= 0 and out[i - 1] > 0.0:
            out[i] = out[i - 1]
            continue
        for nn in range(i + 1, n):
            if out[nn] > 0.0:
                out[i] = out[nn]
                break
        else:
            raise RuntimeError(f"sigma2_noise[{i}] is non-positive with no positive neighbour")
    return out


def _relion_resize_map(image: np.ndarray, new_size: int) -> np.ndarray:
    """RELION ``resizeMap`` (fftw.cpp:1206): Fourier window of a square image to ``new_size``.

    Forward transform scaled by ``1/N**2``, ``windowFourierTransform`` (fftw.h:809; on
    enlarging only coefficients with ``k**2 <= (N/2)**2`` are kept), unscaled inverse.
    """
    old_size = image.shape[-1]
    old_half, new_half = old_size // 2 + 1, new_size // 2 + 1
    F = np.fft.rfft2(image) / (old_size * old_size)
    out = np.zeros((new_size, new_half), dtype=np.complex128)
    if new_half > old_half:
        ip = np.where(np.arange(old_size) < old_half, np.arange(old_size), np.arange(old_size) - old_size)
        keep = ip[:, None] ** 2 + np.arange(old_half)[None, :] ** 2 <= (old_half - 1) ** 2
        rows, cols = np.nonzero(keep)
        out[ip[rows] % new_size, cols] = F[rows, cols]
    else:
        ip = np.where(np.arange(new_size) < new_half, np.arange(new_size), np.arange(new_size) - new_size)
        out[:, :] = F[ip % old_size, :new_half]
    return np.fft.irfft2(out, s=(new_size, new_size)) * (new_size * new_size)


def _rescale_to_model_grid(image: np.ndarray, pixel_size: float, model_pixel_size: float, ori_size: int) -> np.ndarray:
    """Bring one optics group's image onto the model grid (ml_optimiser.cpp:2934-2955).

    Resize to the model pixel size when the pixel sizes differ by more than 1e-4 A
    (new box ``ROUND(box * pixel / model_pixel)``, made even), then window the
    centred box to ``ori_size`` (Xmipp origin; zero padding when enlarging).
    """
    if abs(float(pixel_size) - float(model_pixel_size)) > 0.0001:
        new_size = int(image.shape[-1] * (float(pixel_size) / float(model_pixel_size)) + 0.5)
        new_size += new_size % 2
        image = _relion_resize_map(image, new_size)
    size = image.shape[-1]
    if size == ori_size:
        return image
    out = np.zeros((ori_size, ori_size), dtype=image.dtype)
    # Physical index p of the new box is logical p - ori_size//2, i.e. p - ori_size//2 + size//2 here.
    lo = max(0, ori_size // 2 - size // 2)
    hi = min(ori_size, ori_size // 2 - size // 2 + size)
    src = slice(lo - ori_size // 2 + size // 2, hi - ori_size // 2 + size // 2)
    out[lo:hi, lo:hi] = image[src, src]
    return out


def compute_avg_unaligned_and_sigma2(
    image_iter: Iterator[Tuple[int, np.ndarray]],
    *,
    ori_size: int,
    pixel_size: float,
    particle_diameter_ang: float,
    width_mask_edge_px: int,
    do_zero_mask: bool,
    nr_optics_groups: int,
    minimum_nr_particles: int = 1000,
    group_pixel_sizes=None,
    model_pixel_size: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """``calculateSumOfPowerSpectra`` + ``setSigmaNoiseEstimates`` (per-group cap defaults to RELION's 1000).

    With ``group_pixel_sizes`` (one per optics group, groups on other pixel sizes or
    boxes) each image is masked with its own group's pixel size and brought onto the
    model grid (``_rescale_to_model_grid``) before it enters the sums, as RELION does.
    """
    n_shells = ori_size // 2 + 1

    Mavg = np.zeros((ori_size, ori_size), dtype=np.float64)
    sum_sigma2 = np.zeros((nr_optics_groups, n_shells), dtype=np.float64)
    sumw = np.zeros(nr_optics_groups, dtype=np.float64)
    radius_px = particle_diameter_ang / (2.0 * pixel_size)

    # Count per-group to respect the 1000-particle cap
    per_group_done = np.zeros(nr_optics_groups, dtype=np.int64)
    total_target = minimum_nr_particles * nr_optics_groups

    total_done = 0
    for opt_grp, img in image_iter:
        if per_group_done[opt_grp] >= minimum_nr_particles:
            continue
        img = img.astype(np.float64, copy=False)
        if group_pixel_sizes is not None:
            my_pixel_size = float(group_pixel_sizes[opt_grp])
            if do_zero_mask:
                img = _softmask_outside_map(img, particle_diameter_ang / (2.0 * my_pixel_size), float(width_mask_edge_px))
            img = _rescale_to_model_grid(img, my_pixel_size, model_pixel_size, ori_size)
        elif img.shape != (ori_size, ori_size):
            raise ValueError(f"image shape {img.shape} != expected {(ori_size, ori_size)}")
        elif do_zero_mask:
            img = _softmask_outside_map(img, radius_px, float(width_mask_edge_px))

        Mavg += img
        ind_spect = _radial_power_spectrum(img, n_shells)
        sum_sigma2[opt_grp] += ind_spect
        sumw[opt_grp] += 1.0
        per_group_done[opt_grp] += 1
        total_done += 1

        if total_done >= total_target:
            break

    total_sum = sumw.sum()
    if total_sum <= 0:
        raise RuntimeError("no particles processed")
    Mavg /= total_sum

    # Power spectrum of the averaged image (divided by 2 for 2-dim complex plane)
    mavg_spect = _radial_power_spectrum(Mavg, n_shells) / 2.0

    sigma2_per_group = np.zeros_like(sum_sigma2)
    for g in range(nr_optics_groups):
        if sumw[g] <= 0:
            continue
        sigma2_per_group[g] = sum_sigma2[g] / (2.0 * sumw[g]) - mavg_spect
        sigma2_per_group[g] = _fix_negative_sigma2(sigma2_per_group[g])

    return Mavg, sigma2_per_group


def _image_sigma2_iter(
    dataset,
    image_indices: np.ndarray,
    optics_group_by_particle: np.ndarray,
    *,
    batch_size: int,
) -> Iterable[tuple[int, np.ndarray]]:
    for batch_images, _particle_indices, local_indices in dataset.image_source.iter_batches(
        batch_size=batch_size,
        batch_mode="images",
        subset_indices=np.asarray(image_indices, dtype=np.int64),
    ):
        batch_images = np.asarray(batch_images)
        local_indices = np.asarray(local_indices, dtype=np.int64).reshape(-1)
        for image, local_idx in zip(batch_images, local_indices):
            yield int(optics_group_by_particle[int(local_idx)]), image


def read_relion_sigma2_noise_by_group(model, *, context):
    """Every optics group's RELION ``sigma2_noise`` (``MlModel::sigma2_noise[optics_group]``), ``[G, n]``.

    One ``model_optics_group_<g>`` table per group, in group order; ``None`` when the model has none.
    """
    if not isinstance(model, dict):
        return None
    groups = sorted(
        int(match.group(1))
        for key, table in model.items()
        if (match := re.fullmatch(r"model_optics_group_(\d+)", str(key)))
        and hasattr(table, "columns")
        and "rlnSigma2Noise" in table.columns
    )
    if not groups:
        return None
    if groups != list(range(1, len(groups) + 1)):
        raise ValueError(f"{context}: optics-group noise tables must be numbered 1..G, got {groups}")
    return np.stack([np.asarray(model[f"model_optics_group_{g}"]["rlnSigma2Noise"], dtype=np.float64) for g in groups])


def read_relion_single_optics_sigma2_noise(model, *, context):
    """The sole optics group's RELION ``sigma2_noise`` spectrum, or ``None``.

    For callers that carry one spectrum per random half: they fail closed on
    multi-optics models instead of silently using group 1.
    """
    sigma2 = read_relion_sigma2_noise_by_group(model, context=context)
    if sigma2 is not None and sigma2.shape[0] > 1:
        raise NotImplementedError(
            f"Strict RELION replay does not yet support {sigma2.shape[0]} optics-group sigma2_noise tables in {context}"
        )
    return None if sigma2 is None else sigma2[0]


def relion_mpi_process_start_scoring_noise_pair(noise_half1, noise_half2, *, split_random_halves):
    """Return the noise arrays that RELION MPI uses at process-start scoring.

    AutoRefine reads a model for each random subset, but MPI initialisation
    then calls ``initialiseSigma2Noise`` only on follower rank 1 and broadcasts
    that rank's ``mymodel.sigma2_noise`` to every follower. Consequently both
    random subsets score with the half-1 spectrum at process start. Later
    uninterrupted iterations update each follower independently. Class3D has
    one shared model and does not need this emulation.
    """

    # RELION keeps sigma2_noise in RFLOAT and casts only its reciprocal to
    # XFLOAT when constructing Minvsigma2.  Preserve the caller's dtype here:
    # an early float32 cast changes that reciprocal by one ULP on some shells.
    first = np.asarray(noise_half1)
    second = np.asarray(noise_half2)
    if split_random_halves:
        second = first.copy()
    return [first, second]
