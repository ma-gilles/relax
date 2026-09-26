"""RELION's fine-pass E-step and M-step sums for one K=1 pass, in NumPy (a test reference).

It follows RELION 5.0.1's GPU path (``src/acc/acc_ml_optimiser_impl.h`` and the CUDA kernels in
``src/acc/cuda/cuda_kernels``) for a single-class, 2D-image, 3D-reference pass, from the pass's
inputs: the Fourier images, the CTFs, the noise spectrum, the ``PPref`` projector slab, the fine
rotation and translation grids with their coarse parents, the significant coarse samples and the
priors. Everything is float64 and written as loops over pixels and hypotheses, so it shares no
code with relax's engines. Only :func:`helpers.relion_projector_reference.project` is reused, and
that is itself a port of ``Projector::project`` checked against RELION through ``relion_bind``.

Conventions (all RELION's):

- Images are FFTW half images ``[N, N//2 + 1]``: row ``r`` is ``ky = r`` for ``r <= N/2`` and
  ``r - N`` otherwise (``FFTW_ELEM``), column ``kx = 0..N/2``.
- ``Mresol_fine`` (``MlOptimiser::updateImageSizeAndResolutionPointers``) keeps, on the
  ``current_size`` crop, the pixels with ``ires = round(|k|) <= current_size/2`` and drops the
  ``kx = 0, ky < 0`` half-column that FFTW stores twice.
- Scoring (``cuda_kernel_diff2_fine``): ``diff2 = highres_Xi2/2 + sum 0.5 * |shifted - CTF P|^2 *
  Minvsigma2`` over ``Mresol_fine`` without the origin (``Minvsigma2[0] = 0`` in the E-step).
- Translation (``translatePixel``): ``shifted = X * exp(i (kx a_x + ky a_y))`` with
  ``a = -2 pi t / N``.
- Weights (``convertAllSquaredDifferencesToWeights``): ``w = exp(min diff2 - diff2) * prior``;
  the M-step keeps ``w >= significant_weight``, the sorted weight at
  ``findThresholdIdxInCumulativeSum`` of ``(1 - adaptive_fraction) * sum w``, and divides by
  ``sum w``.
- M-step (``cuda_kernel_backproject3D`` and ``cuda_kernel_wavg``): per pixel of ``Mresol_fine``
  with the origin, ``data += sum_t w CTF Minvsigma2 shifted``, ``weight += sum_t w CTF^2
  Minvsigma2``, spread trilinearly at ``A^-1 k * padding`` inside the BackProjector sphere, with
  the Hermitian mate for ``x < 0``; the noise sums ``sum_t w |CTF P - shifted|^2`` per shell, and
  ``XA``/``AA`` for the scale correction.

Units are relax's: images and noise in RECOVAR units, projections ``-N^2`` times the ``PPref``
projection, as relax feeds the engines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from helpers.relion_projector_reference import project


def fftw_rows(n: int) -> np.ndarray:
    """``ky`` of each FFTW row of an ``n``-row half image."""

    rows = np.arange(n)
    return np.where(rows <= n // 2, rows, rows - n)


def mresol_fine(n: int, current_size: int) -> np.ndarray:
    """RELION's ``Mresol_fine`` on the full ``[n, n//2+1]`` half grid: shell index, -1 when unused."""

    ky = fftw_rows(n)[:, None]
    kx = np.arange(n // 2 + 1)[None, :]
    ires = np.floor(np.sqrt(kx**2 + ky**2) + 0.5).astype(np.int64)
    inside_crop = (ky <= current_size // 2) & (ky > -(current_size // 2)) & (kx <= current_size // 2)
    keep = inside_crop & (ires <= current_size // 2) & ~((kx == 0) & (ky < 0))
    return np.where(keep, ires, -1)


def power_class(image: np.ndarray, current_size: int):
    """``cuda_kernel_powerClass``: shell power of the whole image and the high-resolution tail."""

    n = image.shape[0]
    ky = fftw_rows(n)[:, None]
    kx = np.arange(n // 2 + 1)[None, :] * np.ones((n, 1), dtype=np.int64)
    ires = np.rint(np.sqrt(kx**2 + ky**2)).astype(np.int64)
    use = (ires > 0) & (ires < n // 2 + 1) & ~((kx == 0) & (ky < 0))
    power = np.abs(image) ** 2
    spectrum = np.zeros(n // 2 + 1)
    np.add.at(spectrum, ires[use], power[use])
    highres = float(np.sum(power[use & (ires >= current_size // 2 + 1)]))
    return spectrum, highres


@dataclass
class Pass:
    """One K=1 fine pass: the engine's inputs in RELION layouts and relax units."""

    images: np.ndarray  # complex [n_images, N, N//2+1]
    ctf: np.ndarray  # real [n_images, N, N//2+1]
    sigma2: np.ndarray  # real [N, N//2+1] noise variance per pixel
    projector: dict  # {"data", "pad", "r_max"}: the PPref slab
    fine_rotations: np.ndarray  # [R, 3, 3]
    rotation_parent: np.ndarray  # [R] coarse rotation of each fine rotation
    fine_translations: np.ndarray  # [T, 2] pixels (x, y)
    translation_parent: np.ndarray  # [T] coarse translation of each fine translation
    coarse_support: list  # per image: sorted coarse cell ids (rot * n_coarse_trans + trans) or None
    rotation_log_prior: np.ndarray  # [n_coarse_rot]
    translation_log_prior: np.ndarray | None  # [n_images, n_coarse_trans]
    current_size: int
    reconstruction_padding: int
    adaptive_fraction: float = 0.999
    group_ids: np.ndarray | None = None
    scale: np.ndarray | None = None  # per image
    data_vs_prior: np.ndarray | None = None  # [n_shells], scale sums use shells with > 3
    # K > 1: one ClassModel per class, which then replaces projector, coarse_support,
    # rotation_log_prior (the class's log pdf_class included) and data_vs_prior.
    classes: list | None = None
    # Local search: each image's explicit (fine rotation, fine translation) candidates, the
    # log prior of each fine rotation and each image's fine translation log prior
    # ``[n_images, T]``. They replace coarse_support and the parent-broadcast priors.
    image_cells: list | None = None
    fine_rotation_log_prior: np.ndarray | None = None
    fine_translation_log_prior: np.ndarray | None = None
    # The masked images that RELION scores and runs the Wavg sums on (``Fimg``); ``images``
    # stay the unmasked ones it backprojects (``Fimg_nomask``). None: the same images.
    masked_images: np.ndarray | None = None
    # ``--grad`` (VDAM): backproject the residual ``shifted - CTF P`` (ml_optimiser.cpp:10092-10105).
    subtract_reference: bool = False
    # Zero oversampling reuses the coarse pass's float32 sum of weights (acc_ml_optimiser_impl.h:2868)
    # and keeps every hypothesis (significant weight = the smallest, :3590).
    coarse_sum_weight: np.ndarray | None = None
    # A point group other than C1 is applied to each BPref after the x=0 enforcement, as
    # symmetriseReconstructions does (ml_optimiser.cpp:5541-5575), by RELION's own
    # BackProjector::applyPointGroupSymmetry through relion_bind.
    symmetry: str = "C1"


@dataclass
class ClassModel:
    projector: dict
    coarse_support: list
    rotation_log_prior: np.ndarray
    data_vs_prior: np.ndarray | None = None


def class_models(p: Pass) -> list:
    """The pass's classes; a K=1 pass is its own single class."""

    if p.classes is not None:
        return list(p.classes)
    return [ClassModel(p.projector, p.coarse_support, p.rotation_log_prior, p.data_vs_prior)]


@dataclass
class Posterior:
    cells: np.ndarray  # [M, 3] (class, fine rotation, fine translation); joint over classes
    diff2: np.ndarray  # [M]
    log_weight: np.ndarray  # [M] -diff2 + log priors
    weight: np.ndarray  # [M] normalized posterior
    kept: np.ndarray  # [M] bool, >= RELION's significant weight


def _cells(p: Pass, model: ClassModel, image: int) -> np.ndarray:
    if p.image_cells is not None:
        return np.asarray(p.image_cells[image], dtype=np.int64)
    n_coarse_trans = int(np.max(p.translation_parent)) + 1
    n_coarse = int(model.rotation_log_prior.shape[0]) * n_coarse_trans
    support = model.coarse_support[image]
    coarse = np.arange(n_coarse) if support is None else np.asarray(support, dtype=np.int64)
    cells = []
    for cell in coarse:
        rot_parent, trans_parent = divmod(int(cell), n_coarse_trans)
        for rot in np.flatnonzero(p.rotation_parent == rot_parent):
            for trans in np.flatnonzero(p.translation_parent == trans_parent):
                cells.append((rot, trans))
    return np.asarray(cells, dtype=np.int64)


def _shift_table(p: Pass) -> np.ndarray:
    """``exp(i (kx a_x + ky a_y))`` for every fine translation: ``[T, N, N//2+1]``."""

    n = p.images.shape[1]
    ky = fftw_rows(n)[:, None]
    kx = np.arange(n // 2 + 1)[None, :]
    angles = -2.0 * np.pi * np.asarray(p.fine_translations, dtype=np.float64) / n
    return np.exp(1j * (kx[None] * angles[:, 0, None, None] + ky[None] * angles[:, 1, None, None]))


def projections(p: Pass, projector: dict | None = None) -> np.ndarray:
    """Relax-unit projections ``-N^2 P`` of every fine rotation: ``[R, N, N//2+1]``.

    RELION projects into the ``current_size`` crop (``AccProjectorKernel::makeKernel`` with the
    image's ``maxR``), so a pixel of shell ``current_size/2`` whose radius exceeds
    ``current_size/2`` is zero. A pixel whose rotated radius is exactly the projector's ``r_max``
    is inside in exact arithmetic; the reference keeps it, so the rounding of a float32 rotation
    matrix (about 1e-7 in the rotated radius) cannot drop it.
    """

    n = p.images.shape[1]
    size = int(p.current_size)
    projector = p.projector if projector is None else projector
    projector = dict(projector, r_max=float(projector["r_max"]) * (1.0 + 1e-6))
    cropped = project(projector, np.asarray(p.fine_rotations, dtype=np.float64), size)
    out = np.zeros((cropped.shape[0], n, n // 2 + 1), dtype=np.complex128)
    rows = fftw_rows(size) % n
    out[:, rows, : size // 2 + 1] = cropped
    return -(n**2) * out


def significant_weight(weights: np.ndarray, adaptive_fraction: float) -> float:
    """RELION's fine-pass ``significant_weight`` (ascending sort, cumulative-sum crossing)."""

    ordered = np.sort(weights)
    cumulative = np.cumsum(ordered)
    threshold = (1.0 - adaptive_fraction) * cumulative[-1]
    index = 0
    for i in range(len(cumulative) - 1):
        if cumulative[i] <= threshold < cumulative[i + 1]:
            index = i + 1
    return float(ordered[index])


def posteriors(p: Pass) -> list[Posterior]:
    """Each image's joint posterior over its (class, rotation, translation) candidates."""

    n = p.images.shape[1]
    models = class_models(p)
    projs = [projections(p, model.projector) for model in models]
    shifts = _shift_table(p)
    resol = mresol_fine(n, p.current_size)
    score_pixels = (resol > 0).astype(np.float64)  # the origin is out of the E-step
    inv_sigma2 = 1.0 / p.sigma2
    out = []
    for i in range(p.images.shape[0]):
        scale = 1.0 if p.scale is None else float(p.scale[i])
        ctf = p.ctf[i] * scale
        scored = p.images if p.masked_images is None else p.masked_images
        _, highres = power_class(scored[i], p.current_size)
        highres_native = highres / float(n) ** 4
        shifted = scored[i][None] * shifts
        cells, diff2, log_weight = [], [], []
        for k, model in enumerate(models):
            class_cells = _cells(p, model, i)
            diff = shifted[class_cells[:, 1]] - ctf[None] * projs[k][class_cells[:, 0]]
            class_diff2 = 0.5 * np.sum(np.abs(diff) ** 2 * inv_sigma2 * score_pixels, axis=(1, 2))
            class_diff2 = class_diff2 + 0.5 * highres_native
            if p.fine_rotation_log_prior is not None:
                rotation_prior = p.fine_rotation_log_prior[class_cells[:, 0]]
            else:
                rotation_prior = model.rotation_log_prior[p.rotation_parent[class_cells[:, 0]]]
            class_log_weight = -class_diff2 + rotation_prior
            if p.fine_translation_log_prior is not None:
                class_log_weight = class_log_weight + p.fine_translation_log_prior[i, class_cells[:, 1]]
            elif p.translation_log_prior is not None:
                class_log_weight = (
                    class_log_weight + p.translation_log_prior[i, p.translation_parent[class_cells[:, 1]]]
                )
            cells.append(np.column_stack([np.full(len(class_cells), k), class_cells]))
            diff2.append(class_diff2)
            log_weight.append(class_log_weight)
        cells, diff2, log_weight = np.concatenate(cells), np.concatenate(diff2), np.concatenate(log_weight)
        if p.coarse_sum_weight is not None:
            # The fine weights keep their own exponent shift, exp(score - max + 50)
            # (kernel_exponentiate, acc_ml_optimiser_impl.h:3497-3500), over the coarse sum.
            weights = np.exp(log_weight - log_weight.max() + 50.0)
            out.append(Posterior(cells, diff2, log_weight, weights / float(p.coarse_sum_weight[i]), weights >= 0))
            continue
        weights = np.exp(log_weight - log_weight.max())
        kept = weights >= significant_weight(weights, p.adaptive_fraction)
        out.append(Posterior(cells, diff2, log_weight, weights / weights.sum(), kept))
    return out


def rotation_posterior_sums(p: Pass, posts: list[Posterior], class_index: int = 0) -> np.ndarray:
    """One class's retained (M-step) posterior mass per coarse rotation, summed over images."""

    sums = np.zeros(int(class_models(p)[class_index].rotation_log_prior.shape[0]))
    for post in posts:
        use = post.kept & (post.cells[:, 0] == class_index)
        np.add.at(sums, p.rotation_parent[post.cells[use, 1]], post.weight[use])
    return sums


def backprojector_shape(p: Pass) -> tuple:
    """RELION's BackProjector data shape ``(z, y, x_half)`` for the reconstruction size."""

    r_max = p.current_size // 2
    pad_size = 2 * (int(p.reconstruction_padding * r_max + 0.5) + 1) + 1
    return (pad_size, pad_size, pad_size // 2 + 1)


def mstep(p: Pass, posts: list[Posterior]) -> dict:
    """Per-class BPref data and weight (RELION layout) and the pass's noise, norm and scale sums.

    ``data`` and ``weight`` are the first class's arrays; ``class_data`` and ``class_weight``
    hold every class. The noise and norm sums are one total over classes (RELION's
    ``wdiff2s`` sum block); the scale sums take each class's own ``data_vs_prior > 3`` mask.
    """

    n = p.images.shape[1]
    models = class_models(p)
    projs = [projections(p, model.projector) for model in models]
    shifts = _shift_table(p)
    resol = mresol_fine(n, p.current_size)
    m_pixels = resol >= 0
    # The M-step keeps the origin (acc_ml_optimiser_impl.h:3881) and zeroes Minvsigma2 off Mresol_fine.
    inv_sigma2 = np.where(m_pixels, 1.0 / p.sigma2, 0.0)
    n_shells = n // 2 + 1
    shape = backprojector_shape(p)
    class_data = [np.zeros(shape, dtype=np.complex128) for _ in models]
    class_weight = [np.zeros(shape, dtype=np.float64) for _ in models]
    r_max = p.current_size // 2
    pad = float(p.reconstruction_padding)
    start = -(shape[0] // 2)
    ky_grid = fftw_rows(n)[:, None] * np.ones((1, n // 2 + 1))
    kx_grid = np.arange(n // 2 + 1)[None, :] * np.ones((n, 1))
    wsum_noise = np.zeros(n_shells)
    wsum_power = np.zeros(n_shells)
    norm_correction = np.zeros(p.images.shape[0])
    n_groups = 1 if p.group_ids is None else int(np.max(p.group_ids)) + 1
    scale_xa = np.zeros(n_groups)
    scale_aa = np.zeros(n_groups)
    class_sumw = np.zeros(len(models))
    for i, post in enumerate(posts):
        scale = 1.0 if p.scale is None else float(p.scale[i])
        ctf = p.ctf[i] * scale
        kept = np.flatnonzero(post.kept)
        # The BPref image per (class, rotation): its kept translations summed (cuda_kernel_backproject3D).
        by_rotation: dict[tuple, list[int]] = {}
        for k in kept:
            by_rotation.setdefault((int(post.cells[k, 0]), int(post.cells[k, 1])), []).append(int(k))
        diff2_pixels = np.zeros((n, n // 2 + 1))
        group = 0 if p.group_ids is None else int(p.group_ids[i])
        for (klass, rot), ks in by_rotation.items():
            data, weight = class_data[klass], class_weight[klass]
            real_image = np.zeros((n, n // 2 + 1), dtype=np.complex128)
            f_weight = np.zeros((n, n // 2 + 1))
            xa_pixels = np.zeros((n, n // 2 + 1))
            aa_pixels = np.zeros((n, n // 2 + 1))
            reference = ctf * projs[klass][rot]
            if p.subtract_reference:
                real_image -= float(post.weight[ks].sum()) * ctf * inv_sigma2 * reference
            for k in ks:
                w = post.weight[k]
                shifted = p.images[i] * shifts[post.cells[k, 2]]
                real_image += w * ctf * inv_sigma2 * shifted
                f_weight += w * ctf * ctf * inv_sigma2
                scored = shifted if p.masked_images is None else p.masked_images[i] * shifts[post.cells[k, 2]]
                diff2_pixels += w * np.abs(reference - scored) ** 2
                xa_pixels += w * np.real(reference * np.conj(scored))
                aa_pixels += w * np.abs(reference) ** 2
            model = models[klass]
            dvp = np.full(n_shells, 4.0) if model.data_vs_prior is None else np.asarray(model.data_vs_prior)
            scaled = m_pixels & (dvp[np.maximum(resol, 0)] > 3.0)
            scale_xa[group] += float(xa_pixels[scaled].sum()) / scale
            scale_aa[group] += float(aa_pixels[scaled].sum()) / scale**2
            a_inv = np.linalg.inv(np.asarray(p.fine_rotations[rot], dtype=np.float64))
            for row, col in zip(*np.nonzero(f_weight > 0)):
                y, x = ky_grid[row, col], kx_grid[row, col]
                xp, yp, zp = (
                    (a_inv[0, 0] * x + a_inv[0, 1] * y) * pad,
                    (a_inv[1, 0] * x + a_inv[1, 1] * y) * pad,
                    (a_inv[2, 0] * x + a_inv[2, 1] * y) * pad,
                )
                if xp * xp + yp * yp + zp * zp > r_max * r_max * pad * pad:
                    continue
                value = real_image[row, col]
                if xp < 0:
                    xp, yp, zp, value = -xp, -yp, -zp, np.conj(value)
                x0, y0, z0 = int(np.floor(xp)), int(np.floor(yp)), int(np.floor(zp))
                fx, fy, fz = xp - x0, yp - y0, zp - z0
                y0 -= start
                z0 -= start
                for dz, wz in ((0, 1 - fz), (1, fz)):
                    for dy, wy in ((0, 1 - fy), (1, fy)):
                        for dx, wx in ((0, 1 - fx), (1, fx)):
                            data[z0 + dz, y0 + dy, x0 + dx] += wz * wy * wx * value
                            weight[z0 + dz, y0 + dy, x0 + dx] += wz * wy * wx * f_weight[row, col]
            class_sumw[klass] += float(post.weight[ks].sum())
        np.add.at(wsum_noise, resol[m_pixels], diff2_pixels[m_pixels])
        norm_correction[i] = float(diff2_pixels[m_pixels].sum())
        spectrum, _ = power_class(p.images[i] if p.masked_images is None else p.masked_images[i], p.current_size)
        for shell in range(p.current_size // 2 + 1, n_shells):
            wsum_noise[shell] += spectrum[shell]
            wsum_power[shell] += spectrum[shell]
            norm_correction[i] += spectrum[shell]
    for k, (data, weight) in enumerate(zip(class_data, class_weight)):
        enforce_hermitian_x0(data, weight)
        if p.symmetry != "C1":
            from relax.relion_bind import _relion_bind_core as bind

            class_data[k], class_weight[k] = (
                np.asarray(a)
                for a in bind.apply_point_group_symmetry_to_bpref(
                    data,
                    weight,
                    p.symmetry,
                    ori_size=n,
                    padding_factor=p.reconstruction_padding,
                    current_size=p.current_size,
                    enforce_hermitian=False,
                )
            )
    return {
        "data": class_data[0],
        "weight": class_weight[0],
        "class_data": class_data,
        "class_weight": class_weight,
        "class_sumw": class_sumw,
        "wsum_sigma2_noise": wsum_noise,
        "wsum_img_power": wsum_power,
        "wsum_norm_correction": norm_correction,
        "wsum_scale_correction_xa": scale_xa,
        "wsum_scale_correction_aa": scale_aa,
        "sumw": float(class_sumw.sum()),
    }


def enforce_hermitian_x0(data: np.ndarray, weight: np.ndarray) -> None:
    """``BackProjector::enforceHermitianSymmetry``: sum each x=0 point with its mate, in place."""

    c = data.shape[0] // 2
    for iz in range(-c, c + 1):
        for iy in range(0 if iz < 0 else 1, c + 1):
            a, b = (iz + c, iy + c), (-iz + c, -iy + c)
            fsum = data[a[0], a[1], 0] + np.conj(data[b[0], b[1], 0])
            data[a[0], a[1], 0], data[b[0], b[1], 0] = fsum, np.conj(fsum)
            wsum = weight[a[0], a[1], 0] + weight[b[0], b[1], 0]
            weight[a[0], a[1], 0], weight[b[0], b[1], 0] = wsum, wsum
