"""RELION's expected angular and translational accuracy, in NumPy float64.

Ports the trial loop of ``MlOptimiser::calculateExpectedAngularErrors``
(RELION 5.0.1 ml_optimiser.cpp:9300-9648) for 3-D references projected into
2-D images, including subtomograms over their tilt images. For every trial
particle and every class with ``pdf_class >= 0.01`` it grows an angular (then a
translational) error along RELION's step schedule until the signal-to-noise of
the difference between the reference projection and its perturbed copy exceeds
the ``pvalue`` 4.60517, and averages the errors over the trials.

The arithmetic keeps RELION's host double operations and their order: the CPU
``Projector::project`` (projector.cpp:630-790), ``shiftImageInFourierTransform``
(fftw.cpp:842-906) and the per-image serial SNR sum. Each step reseeds RELION's
generator with ``random_seed + part_id``, so a trial draws the same numbers at
every step; they are drawn once here (:mod:`relax.helpers.relion_random`).
``relax.relion_bind`` is the unit-test oracle
(``tests/unit/test_relion_expected_accuracy_vs_relion_bind.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from relax.helpers import relion_random

PVALUE = 4.60517
_PI = 3.14159265358979323846
_EQUAL_ACCURACY = 1e-6


@dataclass(frozen=True)
class ExpectedErrors:
    """Per-class and overall accuracies as ``calculateExpectedAngularErrors`` sets them."""

    acc_rot: float
    acc_trans: float
    acc_rot_class: np.ndarray
    acc_trans_class: np.ndarray
    class_counts: np.ndarray


def _relion_round(values):
    values = np.asarray(values, dtype=np.float64)
    return np.where(values > 0, np.trunc(values + 0.5), np.trunc(values - 0.5)).astype(np.int64)


def _euler_matrices(eulers_deg):
    from relax.healpix_sampling import euler_angles_to_matrix

    return euler_angles_to_matrix(eulers_deg)


def _matmul3(left, right):
    """``Matrix2D::operator*`` for ``[..., 3, 3]`` stacks."""

    return (
        left[..., :, 0, None] * right[..., None, 0, :]
        + left[..., :, 1, None] * right[..., None, 1, :]
        + left[..., :, 2, None] * right[..., None, 2, :]
    )


def _inverse3(matrix):
    """``Matrix2D::inv`` 3x3 branch (matrix2d.h:1108-1123): cofactors over the determinant."""

    m = matrix
    inverse = np.empty_like(m)
    inverse[..., 0, 0] = m[..., 2, 2] * m[..., 1, 1] - m[..., 2, 1] * m[..., 1, 2]
    inverse[..., 0, 1] = -(m[..., 2, 2] * m[..., 0, 1] - m[..., 2, 1] * m[..., 0, 2])
    inverse[..., 0, 2] = m[..., 1, 2] * m[..., 0, 1] - m[..., 1, 1] * m[..., 0, 2]
    inverse[..., 1, 0] = -(m[..., 2, 2] * m[..., 1, 0] - m[..., 2, 0] * m[..., 1, 2])
    inverse[..., 1, 1] = m[..., 2, 2] * m[..., 0, 0] - m[..., 2, 0] * m[..., 0, 2]
    inverse[..., 1, 2] = -(m[..., 1, 2] * m[..., 0, 0] - m[..., 1, 0] * m[..., 0, 2])
    inverse[..., 2, 0] = m[..., 2, 1] * m[..., 1, 0] - m[..., 2, 0] * m[..., 1, 1]
    inverse[..., 2, 1] = -(m[..., 2, 1] * m[..., 0, 0] - m[..., 2, 0] * m[..., 0, 1])
    inverse[..., 2, 2] = m[..., 1, 1] * m[..., 0, 0] - m[..., 1, 0] * m[..., 0, 1]
    determinant = (
        m[..., 0, 0] * inverse[..., 0, 0] + m[..., 1, 0] * inverse[..., 0, 1] + m[..., 2, 0] * inverse[..., 0, 2]
    )
    return inverse / determinant[..., None, None]


class _Projector:
    """A ``Projector`` of 3-D ``data`` (``[L, L, L // 2 + 1]``, Xmipp origin) at ``r_max``."""

    def __init__(self, data, r_max: int, padding_factor: int, image_size: int):
        self.data = np.ascontiguousarray(data, dtype=np.complex128)
        self.real = np.ascontiguousarray(self.data.real)
        self.imag = np.ascontiguousarray(self.data.imag)
        self.padding_factor = float(padding_factor)
        self.r_max_ref_2 = float(int(r_max * self.padding_factor) ** 2)
        size = int(image_size)
        half = size // 2 + 1
        r_max_out = half - 1
        rows = np.arange(size)
        y = np.where(rows <= r_max_out, rows, rows - size)
        x_max = np.floor(np.sqrt((r_max_out * r_max_out - y * y).astype(np.float64))).astype(np.int64)
        # The pixels Projector::project visits: row i, x in [0, FLOOR(sqrt(r_max_out^2 - y^2))].
        row_index, x_index = np.nonzero(np.arange(half)[None, :] <= x_max[:, None])
        self.pixel_flat = row_index * half + x_index
        self.pixel_x = x_index.astype(np.float64)
        self.pixel_y = y[row_index].astype(np.float64)
        self.shape = (size, half)

    def project(self, matrices):
        """``get2DFourierTransform(F, A)`` for ``[M, 3, 3]`` ``A``; returns ``[M, H, W]`` real and imag."""

        a_inv = _inverse3(np.asarray(matrices, dtype=np.float64)) * self.padding_factor
        x, y = self.pixel_x[None, :], self.pixel_y[None, :]
        xp = a_inv[:, 0, 0, None] * x + a_inv[:, 0, 1, None] * y
        yp = a_inv[:, 1, 0, None] * x + a_inv[:, 1, 1, None] * y
        zp = a_inv[:, 2, 0, None] * x + a_inv[:, 2, 1, None] * y
        inside = (xp * xp + yp * yp + zp * zp) <= self.r_max_ref_2
        negative = xp < 0
        xp = np.where(negative, -xp, xp)
        yp = np.where(negative, -yp, yp)
        zp = np.where(negative, -zp, zp)
        x0 = np.floor(xp)
        y0 = np.floor(yp)
        z0 = np.floor(zp)
        fx, fy, fz = xp - x0, yp - y0, zp - z0
        nz, ny, nx = self.data.shape
        x0 = x0.astype(np.int64)
        y0 = y0.astype(np.int64) + ny // 2
        z0 = z0.astype(np.int64) + nz // 2
        inside &= (x0 >= 0) & (x0 + 1 < nx) & (y0 >= 0) & (y0 + 1 < ny) & (z0 >= 0) & (z0 + 1 < nz)
        x0 = np.where(inside, x0, 0)
        y0 = np.where(inside, y0, 0)
        z0 = np.where(inside, z0, 0)

        def interpolate(values):
            d000, d001 = values[z0, y0, x0], values[z0, y0, x0 + 1]
            d010, d011 = values[z0, y0 + 1, x0], values[z0, y0 + 1, x0 + 1]
            d100, d101 = values[z0 + 1, y0, x0], values[z0 + 1, y0, x0 + 1]
            d110, d111 = values[z0 + 1, y0 + 1, x0], values[z0 + 1, y0 + 1, x0 + 1]
            # LIN_INTERP(a, l, h) = l + (h - l) * a, nested as projector.cpp:733-740.
            dx00 = d000 + (d001 - d000) * fx
            dx01 = d100 + (d101 - d100) * fx
            dx10 = d010 + (d011 - d010) * fx
            dx11 = d110 + (d111 - d110) * fx
            dxy0 = dx00 + (dx10 - dx00) * fy
            dxy1 = dx01 + (dx11 - dx01) * fy
            return dxy0 + (dxy1 - dxy0) * fz

        real = np.where(inside, interpolate(self.real), 0.0)
        imag = interpolate(self.imag)
        imag = np.where(inside, np.where(negative, -imag, imag), 0.0)
        out_real = np.zeros((matrices.shape[0], self.shape[0] * self.shape[1]))
        out_imag = np.zeros_like(out_real)
        out_real[:, self.pixel_flat] = real
        out_imag[:, self.pixel_flat] = imag
        return out_real.reshape(-1, *self.shape), out_imag.reshape(-1, *self.shape)


def _shift(real, imag, oridim, xshift, yshift):
    """``shiftImageInFourierTransform(in, out, oridim, xshift, yshift)`` in 2-D, per image."""

    size, half = real.shape[-2:]
    xs = np.asarray(xshift, dtype=np.float64) / -float(oridim)
    ys = np.asarray(yshift, dtype=np.float64) / -float(oridim)
    unchanged = (np.abs(xs) < _EQUAL_ACCURACY) & (np.abs(ys) < _EQUAL_ACCURACY)
    rows = np.arange(size)
    y = np.where(rows < half, rows, rows - size).astype(np.float64)
    x = np.arange(half, dtype=np.float64)
    dotp = 2 * _PI * (x[None, None, :] * xs[:, None, None] + y[None, :, None] * ys[:, None, None])
    b, a = np.sin(dotp), np.cos(dotp)
    ac = a * real
    bd = b * imag
    ab_cd = (a + b) * (real + imag)
    out_real = np.where(unchanged[:, None, None], real, ac - bd)
    out_imag = np.where(unchanged[:, None, None], imag, ab_cd - ac - bd)
    return out_real, out_imag


def _schedule(mode: int) -> tuple[np.ndarray, np.ndarray, int]:
    """RELION's error steps: ``(ang_errors, sh_errors, n_scored)`` up to the mode's cap."""

    ang, sh = 0.0, 0.0
    angles, shifts = [], []
    while True:
        ang_step = (
            0.05
            if ang < 0.2
            else 0.1
            if ang < 1.0
            else 0.2
            if ang < 2.0
            else 0.5
            if ang < 5.0
            else (1.0 if ang < 10.0 else 2.0 if ang < 20.0 else 5.0)
        )
        sh_step = 0.1 if sh < 1.0 else 0.2 if sh < 2.0 else 0.5 if sh < 5.0 else 1.0 if sh < 10.0 else 2.0
        ang += ang_step
        sh += sh_step
        angles.append(ang)
        shifts.append(sh)
        if (mode == 0 and ang > 30.0) or (mode == 1 and sh > 10.0):
            return np.asarray(angles), np.asarray(shifts), len(angles) - 1


def _snr_terms(image_size, full_size, sigma2_noise, sigma2_fudge, remap_image_sizes):
    """Per-pixel ``1 / (2 fudge sigma2)`` of the pixels the SNR sums, zero elsewhere."""

    half = image_size // 2 + 1
    rows = np.arange(image_size)
    iy = np.where(rows < half, rows, rows - image_size)[:, None]
    ix = np.arange(half)[None, :]
    ires = _relion_round(np.sqrt((iy * iy + ix * ix).astype(np.float64)))
    remapped = _relion_round(remap_image_sizes * ires)
    sigma = np.asarray(sigma2_noise, dtype=np.float64)
    valid = (ires > 0) & (ires < half) & ~((ix == 0) & (iy < 0)) & (remapped < sigma.size)
    shell_sigma = np.where(valid, sigma[np.minimum(remapped, sigma.size - 1)], 0.0)
    valid &= shell_sigma > 0.0
    denominator = np.where(valid, 2.0 * sigma2_fudge * shell_sigma, 1.0)
    return valid, denominator


def _serial_snr(diff_real, diff_imag, valid, denominator, image_offsets):
    """``my_snr += norm(F1 - F2) / (2 fudge sigma)`` over images then pixels, in RELION's order."""

    terms = np.where(valid, (diff_real * diff_real + diff_imag * diff_imag) / denominator, 0.0)
    terms = terms.reshape(terms.shape[0], -1)
    snr = np.empty(image_offsets.size - 1)
    for trial in range(snr.size):
        block = terms[image_offsets[trial] : image_offsets[trial + 1]].reshape(-1)
        snr[trial] = np.cumsum(block)[-1] if block.size else 0.0
    return snr


_CHUNK_VALUES = 1 << 22


def _trial_chunks(counts, max_images: int) -> list[np.ndarray]:
    """Consecutive trial groups holding at most ``max_images`` images (one trial at least)."""

    chunks, current, held = [], [], 0
    for trial, count in enumerate(np.asarray(counts, dtype=np.int64)):
        if current and held + count > max_images:
            chunks.append(np.asarray(current))
            current, held = [], 0
        current.append(trial)
        held += int(count)
    if current:
        chunks.append(np.asarray(current))
    return chunks


def _chunk_errors(projector, matrices, *, rows, counts, eulers, draws, ctf, aproj, image_full_size, valid, denominator):
    """Angular and translational errors of one chunk of trials (``rows`` are their images)."""

    n_trials = counts.size
    image_trial = np.repeat(np.arange(n_trials), counts)
    f1_real, f1_imag = projector.project(matrices(eulers, rows))
    f1_ctf_real, f1_ctf_imag = (f1_real, f1_imag) if ctf is None else (f1_real * ctf, f1_imag * ctf)
    errors = []
    for mode in (0, 1):
        angles, shifts, n_scored = _schedule(mode)
        result = np.full(n_trials, angles[n_scored] if mode == 0 else shifts[n_scored])
        active = np.ones(n_trials, dtype=bool)
        for step in range(n_scored):
            local = np.flatnonzero(active[image_trial])
            if local.size == 0:
                break
            ran = draws[local]
            if mode == 0:
                perturbed = eulers[local].copy()
                column = np.where(ran < 0.3333, 0, np.where(ran < 0.6667, 1, 2))
                perturbed[np.arange(local.size), column] += angles[step]
                f2_real, f2_imag = projector.project(matrices(perturbed, rows[local]))
            else:
                if aproj is not None:
                    # Experiment::getTranslationInTiltSeries (exp_model.cpp:105-113).
                    x3 = np.where(ran < 0.3333, shifts[step], 0.0)
                    y3 = np.where((ran >= 0.3333) & (ran < 0.6667), shifts[step], 0.0)
                    z3 = np.where(ran >= 0.6667, shifts[step], 0.0)
                    a = aproj[local]
                    xshift = a[:, 0, 0] * x3 + a[:, 0, 1] * y3 + a[:, 0, 2] * z3
                    yshift = a[:, 1, 0] * x3 + a[:, 1, 1] * y3 + a[:, 1, 2] * z3
                else:
                    xshift = np.where(ran < 0.5, shifts[step], 0.0)
                    yshift = np.where(ran < 0.5, 0.0, shifts[step])
                f2_real, f2_imag = _shift(f1_real[local], f1_imag[local], image_full_size, -xshift, -yshift)
            if ctf is not None:
                f2_real, f2_imag = f2_real * ctf[local], f2_imag * ctf[local]
            trials = np.flatnonzero(active)
            local_offsets = np.concatenate([[0], np.cumsum(counts[trials])]).astype(np.int64)
            snr = _serial_snr(
                f1_ctf_real[local] - f2_real, f1_ctf_imag[local] - f2_imag, valid, denominator, local_offsets
            )
            # ``while (my_snr <= pvalue)`` stops on anything else, NaN included.
            done = trials[~(snr <= PVALUE)]
            result[done] = angles[step] if mode == 0 else shifts[step]
            active[done] = False
        errors.append(result)
    return errors


def expected_angular_errors(
    *,
    projector_data,
    projector_r_max: int,
    padding_factor: int,
    eulers_deg,
    particle_ids,
    pdf_class,
    sigma2_noise,
    ctf_images,
    pixel_size: float,
    ori_size: int,
    current_image_size: int,
    sigma2_fudge: float,
    random_seed: int,
    random_seed_particle_ids,
    model_pixel_size: float | None = None,
    image_full_size: int | None = None,
    image_offsets=None,
    image_projections=None,
) -> ExpectedErrors:
    """``calculateExpectedAngularErrors`` over the trials, as the binding's oracle computes it.

    ``projector_data`` ``[K, L, L, L // 2 + 1]`` is each class's ``Projector::data``
    at ``projector_r_max``. ``ctf_images`` is ``None`` without CTF correction, else
    one ``[current_image_size, current_image_size // 2 + 1]`` CTF per image of the
    trial particles (in trial-image order). Tilt images: ``image_offsets``
    ``[n_particles + 1]`` and ``image_projections`` ``[n_images, 3, 3]`` (``Aproj``).
    """

    model_pixel_size = float(pixel_size if model_pixel_size is None or model_pixel_size <= 0 else model_pixel_size)
    image_full_size = int(ori_size if image_full_size is None or image_full_size <= 0 else image_full_size)
    scale_difference = (image_full_size * float(pixel_size)) / (int(ori_size) * model_pixel_size)
    remap_image_sizes = (int(ori_size) * model_pixel_size) / (image_full_size * float(pixel_size))
    eulers = np.asarray(eulers_deg, dtype=np.float64).reshape(-1, 3)
    particles = np.asarray(particle_ids, dtype=np.int64).reshape(-1)
    seed_particles = np.asarray(random_seed_particle_ids, dtype=np.int64).reshape(-1)
    pdf = np.asarray(pdf_class, dtype=np.float64).reshape(-1)
    n_classes = pdf.size
    n_trials = eulers.shape[0]
    tomo = image_offsets is not None
    if tomo:
        offsets = np.asarray(image_offsets, dtype=np.int64)
        projections = np.asarray(image_projections, dtype=np.float64)
        first = offsets[particles]
        counts = offsets[particles + 1] - first
        image_rows = (
            np.concatenate([np.arange(f, f + c) for f, c in zip(first, counts, strict=True)])
            if n_trials
            else np.zeros(0, np.int64)
        )
        aproj = projections[image_rows]
    else:
        counts = np.ones(n_trials, dtype=np.int64)
        aproj = None
    trial_image_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    n_images = int(trial_image_offsets[-1])
    image_trial = np.repeat(np.arange(n_trials), counts)

    draws = np.zeros(n_images)
    if np.any(pdf >= 0.01):
        for trial in range(n_trials):
            generator = relion_random.GlibcRand(int(random_seed) + int(seed_particles[trial]))
            for row in range(trial_image_offsets[trial], trial_image_offsets[trial + 1]):
                draws[row] = float(relion_random.rnd_unif(generator))

    valid, denominator = _snr_terms(
        int(current_image_size), image_full_size, sigma2_noise, float(sigma2_fudge), remap_image_sizes
    )
    ctf = None if ctf_images is None else np.asarray(ctf_images, dtype=np.float64)
    image_eulers = eulers[image_trial]

    def matrices(rows_eulers, rows):
        a = _euler_matrices(rows_eulers)
        if tomo:
            a = _matmul3(aproj[rows], a)
        return a * scale_difference

    acc_rot_class = np.full(n_classes, 999.0)
    acc_trans_class = np.full(n_classes, 999.0)
    class_counts = np.zeros(n_classes, dtype=np.int64)
    acc_rot, acc_trans = 999.0, 999.0
    # Trials are independent; a chunk bounds the per-step image arrays.
    n_pixels = int(current_image_size) * (int(current_image_size) // 2 + 1)
    chunks = _trial_chunks(counts, max(1, _CHUNK_VALUES // n_pixels))
    for k in range(n_classes):
        if pdf[k] < 0.01:
            continue
        projector = _Projector(projector_data[k], int(projector_r_max), int(padding_factor), int(current_image_size))
        errors = [np.empty(n_trials), np.empty(n_trials)]
        for chunk in chunks:
            rows = np.concatenate([np.arange(trial_image_offsets[t], trial_image_offsets[t + 1]) for t in chunk])
            chunk_errors = _chunk_errors(
                projector,
                matrices,
                rows=rows,
                counts=counts[chunk],
                eulers=image_eulers[rows],
                draws=draws[rows],
                ctf=None if ctf is None else ctf[rows],
                aproj=None if aproj is None else aproj[rows],
                image_full_size=image_full_size,
                valid=valid,
                denominator=denominator,
            )
            for mode in (0, 1):
                errors[mode][chunk] = chunk_errors[mode]
        rot_sum = 0.0
        trans_sum = 0.0
        for trial in range(n_trials):
            rot_sum += errors[0][trial]
            trans_sum += float(pixel_size) * errors[1][trial]
        if n_trials > 0:
            acc_rot_class[k] = rot_sum / float(n_trials)
            acc_trans_class[k] = trans_sum / float(n_trials)
            class_counts[k] = n_trials
            acc_rot = min(acc_rot, acc_rot_class[k])
            acc_trans = min(acc_trans, acc_trans_class[k])
    return ExpectedErrors(float(acc_rot), float(acc_trans), acc_rot_class, acc_trans_class, class_counts)


__all__ = ["PVALUE", "ExpectedErrors", "expected_angular_errors"]
