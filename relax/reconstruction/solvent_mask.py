"""User reference (solvent) mask and RELION's solvent-corrected gold-standard FSC.

RELION ``--solvent_mask`` replaces the particle-diameter sphere of ``--flatten_solvent``: after each
M-step (not at convergence) every reference is multiplied by the mask (``MlOptimiser::solventFlatten``,
ml_optimiser.cpp:5491-5600). The start-up reference is not masked. ``checkMask`` (:1811-1896) first
brings the mask onto the model grid: resampled when its pixel size differs by more than 0.001 A,
windowed to the model box, clipped to [0, 1].

With ``--solvent_correct_fsc`` (MPI relion_refine only; the non-MPI program ignores the flag) the
half-set FSC is replaced by the phase-randomisation corrected FSC of the masked unregularised half maps
(``reconstructUnregularisedMapAndCalculateSolventCorrectedFSC``, ml_optimiser_mpi.cpp:3221-3403), before
the solvent flatten. See docs/math/relion_refinement_algorithm.md, "Reference mask".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# randomize_at is the first shell whose unmasked FSC falls below this; the corrected formula starts
# two shells above it (RELION: "small artefacts near the resolution of randomisation").
_RANDOMIZE_FSC_AT = 0.8
_CORRECTION_HANDOFF_SHELLS = 2


def _relion_round(values):
    """RELION's ROUND (macros.h): halves away from zero."""
    values = np.asarray(values, dtype=np.float64)
    return np.where(values > 0, np.floor(values + 0.5), np.ceil(values - 0.5)).astype(np.int64)


def _fourier_resize(volume, new_size: int) -> np.ndarray:
    """A cubic map on ``new_size`` voxels by Fourier crop or zero-pad, keeping its real-space values."""

    size = volume.shape[0]
    spectrum = np.fft.fftshift(np.fft.fftn(volume))
    out = np.zeros((new_size,) * 3, dtype=np.complex128)
    keep = min(size, new_size)
    src = slice(size // 2 - keep // 2, size // 2 - keep // 2 + keep)
    dst = slice(new_size // 2 - keep // 2, new_size // 2 - keep // 2 + keep)
    out[dst, dst, dst] = spectrum[src, src, src]
    return np.fft.ifftn(np.fft.ifftshift(out)).real * (new_size / size) ** 3


def _centered_window(volume, box: int) -> np.ndarray:
    """Crop or zero-pad a cubic map about its centre voxel ``N // 2`` (RELION's Xmipp-origin window)."""

    size = volume.shape[0]
    out = np.zeros((box,) * 3, dtype=volume.dtype)
    lo = min(size // 2, box // 2)
    hi = min(size - size // 2, box - box // 2)
    src = slice(size // 2 - lo, size // 2 + hi)
    dst = slice(box // 2 - lo, box // 2 + hi)
    out[dst, dst, dst] = volume[src, src, src]
    return out


def read_solvent_mask(path, *, box: int, pixel_size: float) -> np.ndarray:
    """The user mask on the model grid, float64 ``[box, box, box]`` in the MRC file's (z, y, x) order."""

    import mrcfile

    with mrcfile.open(Path(path), permissive=True) as handle:
        mask = np.asarray(handle.data, dtype=np.float64)
        mask_pixel = float(handle.voxel_size.x)
    if mask.ndim != 3 or len(set(mask.shape)) != 1:
        raise ValueError(f"the solvent mask {path} must be a cubic 3-D map, got shape {mask.shape}")
    if mask_pixel > 0 and abs(mask_pixel - float(pixel_size)) > 0.001:
        new_size = int(_relion_round(mask.shape[0] * mask_pixel / float(pixel_size)))
        new_size += new_size % 2
        mask = _fourier_resize(mask, new_size)
    if mask.shape[0] != int(box):
        mask = _centered_window(mask, int(box))
    return np.clip(mask, 0.0, 1.0)


def _fftw_shells(size: int):
    """Shell label of each rfft voxel, RELION getFSC's ``ROUND(|k|)``; and ``|k|^2``."""

    z = np.fft.fftfreq(size) * size
    x = np.arange(size // 2 + 1)
    radius_sq = z[:, None, None] ** 2 + z[None, :, None] ** 2 + x[None, None, :] ** 2
    return _relion_round(np.sqrt(radius_sq)), radius_sq


def real_space_fsc(map1, map2) -> np.ndarray:
    """RELION ``getFSC`` of two real cubic maps: shells ``ROUND(|k|) < N // 2 + 1``, float64."""

    size = map1.shape[0]
    shells, _ = _fftw_shells(size)
    ft1, ft2 = np.fft.rfftn(map1), np.fft.rfftn(map2)
    n_shells = size // 2 + 1
    inside = shells < n_shells
    labels = shells[inside]
    num = np.bincount(labels, (np.conj(ft1) * ft2).real[inside], minlength=n_shells)
    den1 = np.bincount(labels, (np.abs(ft1) ** 2)[inside], minlength=n_shells)
    den2 = np.bincount(labels, (np.abs(ft2) ** 2)[inside], minlength=n_shells)
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / np.sqrt(den1 * den2)


def randomize_phases_beyond(volume, shell: int, phases) -> np.ndarray:
    """``volume`` with each rfft voxel at ``|k| >= shell`` given phase ``phases`` (same shape as the rfft)."""

    _, radius_sq = _fftw_shells(volume.shape[0])
    spectrum = np.fft.rfftn(volume)
    spectrum = np.where(radius_sq >= shell * shell, np.abs(spectrum) * np.exp(1j * phases), spectrum)
    return np.fft.irfftn(spectrum, s=volume.shape, axes=(0, 1, 2))


def solvent_corrected_fsc(half1, half2, mask, *, current_size: int, rng: np.random.Generator):
    """RELION's solvent-corrected FSC of two unregularised half maps (real space, cubic, float64).

    Returns ``(fsc, details)``. ``fsc`` is the masked FSC below ``randomize_at + 2`` and
    ``(masked - random) / (1 - random)`` (0 where ``random > masked``) above it, ``randomize_at``
    being the first shell above 0 whose unmasked FSC is below 0.8; without such a shell it is the
    unmasked FSC. Shells beyond ``current_size // 2`` are 0. ``rng`` draws the uniform phases in
    [0, 2 pi) that replace both maps' phases beyond ``randomize_at``; RELION draws its own, so the
    corrected shells agree with RELION only statistically.
    """

    half1 = np.asarray(half1, dtype=np.float64)
    half2 = np.asarray(half2, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    if not half1.shape == half2.shape == mask.shape:
        raise ValueError(f"half maps and mask must share a shape, got {half1.shape}, {half2.shape}, {mask.shape}")
    unmasked = real_space_fsc(half1, half2)
    masked = real_space_fsc(half1 * mask, half2 * mask)
    below = np.nonzero(unmasked[1:] < _RANDOMIZE_FSC_AT)[0]
    randomize_at = int(below[0]) + 1 if below.size else -1
    random_masked = None
    if randomize_at > 0:
        spectrum_shape = (half1.shape[0], half1.shape[1], half1.shape[2] // 2 + 1)
        random1 = randomize_phases_beyond(half1, randomize_at, rng.uniform(0.0, 2.0 * np.pi, spectrum_shape))
        random2 = randomize_phases_beyond(half2, randomize_at, rng.uniform(0.0, 2.0 * np.pi, spectrum_shape))
        random_masked = real_space_fsc(random1 * mask, random2 * mask)
        shells = np.arange(masked.size)
        with np.errstate(divide="ignore", invalid="ignore"):
            corrected = np.where(random_masked > masked, 0.0, (masked - random_masked) / (1.0 - random_masked))
        fsc = np.where(shells < randomize_at + _CORRECTION_HANDOFF_SHELLS, masked, corrected)
    else:
        fsc = unmasked.copy()
    fsc[int(current_size) // 2 + 1 :] = 0.0
    details = {"randomize_at": randomize_at, "unmasked": unmasked, "masked": masked, "random_masked": random_masked}
    return fsc, details
