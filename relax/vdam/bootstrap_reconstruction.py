"""RELION's InitialModel bootstrap reference, in NumPy float64.

Ports what ``MlOptimiser`` does before the first VDAM iteration of a de novo
InitialModel (RELION 5.0.1): back-project the first particles at random
orientations into one ``BackProjector`` per class
(``calculateSumOfPowerSpectraAndAverageImage``, ml_optimiser.cpp:3127-3205),
``BackProjector::reconstruct`` without regularisation (skip-gridding path,
backprojector.cpp:1513-2023), then ``initialLowPassFilterReferences``,
``SomGraph::make_blobs_3d`` and the particle-diameter soft mask
(ml_optimiser.cpp:2940-2980).

The host double arithmetic keeps RELION's formulas and operation order; FFTs
(NumPy's pocketfft rather than FFTW) and the order of the back-projection sums
agree with RELION's to float rounding. The C ``rand()`` stream is RELION's
(:mod:`relax.helpers.relion_random`), and the blob draws continue it where the
particle loop leaves it, as in RELION. ``relax.relion_bind`` is the unit-test
oracle (``tests/unit/initial_model/test_bootstrap_reconstruction_vs_relion_bind.py``).
"""

from __future__ import annotations

import math

import numpy as np

from relax.healpix_sampling import euler_angles_to_matrix
from relax.helpers import relion_random
from relax.refinement.mean_helpers import initial_low_pass_filter_references
from relax.vdam.schedules import _relion_round

_PI = 3.14159265358979323846


def _logical_axis(size: int) -> np.ndarray:
    """Xmipp logical indices ``FIRST_XMIPP_INDEX(size) ...`` of one axis."""

    first = -int(float(size) / 2.0)
    return np.arange(first, first + size)


def _fftw_axis(n_full: int, n_half: int) -> np.ndarray:
    """``FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM`` logical index of a full axis (``k < XSIZE ? k : k - N``)."""

    index = np.arange(n_full)
    return np.where(index < n_half, index, index - n_full)


def fftw_window_rows(full_size: int, size: int) -> np.ndarray:
    """Rows of a full FFTW half image that ``windowFourierTransform`` keeps for ``size`` (fftw.h:809-857).

    Row ``i`` of the window holds logical frequency ``i`` for ``i < size // 2 + 1``, else
    ``i - size``; the column window is ``[0, size // 2 + 1)``.
    """

    rows = np.arange(int(size))
    logical = np.where(rows < int(size) // 2 + 1, rows, rows - int(size))
    return np.where(logical < 0, logical + int(full_size), logical)


def soft_mask_outside_map(volume, radius: float = -1.0, cosine_width: float = 3.0) -> np.ndarray:
    """``softMaskOutsideMap(vol, radius, cosine_width)`` without a noise map (mask.cpp).

    The background average is RELION's serial double sum over the logical grid.
    """

    vol = np.asarray(volume, dtype=np.float64)
    axes = [_logical_axis(size) for size in vol.shape]
    grids = np.meshgrid(*axes, indexing="ij")
    r = np.sqrt(sum(g.astype(np.int64) * g for g in grids).astype(np.float64))
    if radius < 0:
        radius = float(vol.shape[-1]) / 2.0
    radius_p = radius + cosine_width
    outside = r > radius_p
    edge = ~outside & ~(r < radius)
    raised = np.where(edge, 0.5 + 0.5 * np.cos(_PI * (radius_p - r) / cosine_width), 0.0)
    weight = np.where(outside, 1.0, raised).reshape(-1)
    weighted = np.where(outside, vol, raised * vol).reshape(-1)
    counted = outside.reshape(-1) | edge.reshape(-1)
    total = np.cumsum(weight[counted])[-1] if np.any(counted) else 0.0
    background = (np.cumsum(weighted[counted])[-1] if np.any(counted) else 0.0) / total
    out = np.where(outside, background, vol)
    return np.where(edge, (1 - raised) * vol + raised * background, out)


def initial_low_pass_filter(volume, ori_size: int, pixel_size: float, ini_high_ang: float) -> np.ndarray:
    """``initialLowPassFilterReferences`` for one reference (ml_optimiser.cpp:3556-3586)."""

    vol = np.asarray(volume, dtype=np.float64)
    if not ini_high_ang > 0.0:
        return vol
    return initial_low_pass_filter_references(
        vol[None], ori_size=int(ori_size), pixel_size=float(pixel_size), ini_high_ang=float(ini_high_ang)
    )[0]


class BackProjector3D:
    """A RELION ``BackProjector`` of a 3-D reference fed with 2-D images (trilinear, no symmetry)."""

    def __init__(self, ori_size: int, padding_factor: int, current_size: int = -1):
        self.ori_size = int(ori_size)
        self.padding_factor = float(padding_factor)
        r_max = self.ori_size // 2 if current_size < 0 else int(current_size) // 2
        self.r_max = min(r_max, self.ori_size // 2)
        self.pad_size = 2 * (_relion_round(self.padding_factor * self.r_max) + 1) + 1
        shape = (self.pad_size, self.pad_size, self.pad_size // 2 + 1)
        self.data = np.zeros(shape, dtype=np.complex128)
        self.weight = np.zeros(shape, dtype=np.float64)

    def backproject(self, images, matrices, weights) -> None:
        """``set2DFourierTransform(F, A, &Mweight)`` for each ``[H, H // 2 + 1]`` image (backprojector.cpp:87-330)."""

        images = np.asarray(images, dtype=np.complex128)
        weights = np.asarray(weights, dtype=np.float64)
        a = np.asarray(matrices, dtype=np.float64)
        s, sh = images.shape[-2:]
        m = a
        # Matrix2D::inv 3x3: cofactors over the determinant, then times padding_factor.
        inv = np.empty_like(m)
        inv[:, 0, 0] = m[:, 2, 2] * m[:, 1, 1] - m[:, 2, 1] * m[:, 1, 2]
        inv[:, 0, 1] = -(m[:, 2, 2] * m[:, 0, 1] - m[:, 2, 1] * m[:, 0, 2])
        inv[:, 0, 2] = m[:, 1, 2] * m[:, 0, 1] - m[:, 1, 1] * m[:, 0, 2]
        inv[:, 1, 0] = -(m[:, 2, 2] * m[:, 1, 0] - m[:, 2, 0] * m[:, 1, 2])
        inv[:, 1, 1] = m[:, 2, 2] * m[:, 0, 0] - m[:, 2, 0] * m[:, 0, 2]
        inv[:, 1, 2] = -(m[:, 1, 2] * m[:, 0, 0] - m[:, 1, 0] * m[:, 0, 2])
        inv[:, 2, 0] = m[:, 2, 1] * m[:, 1, 0] - m[:, 2, 0] * m[:, 1, 1]
        inv[:, 2, 1] = -(m[:, 2, 1] * m[:, 0, 0] - m[:, 2, 0] * m[:, 0, 1])
        inv[:, 2, 2] = m[:, 1, 1] * m[:, 0, 0] - m[:, 1, 0] * m[:, 0, 1]
        det = m[:, 0, 0] * inv[:, 0, 0] + m[:, 1, 0] * inv[:, 0, 1] + m[:, 2, 0] * inv[:, 0, 2]
        inv = inv / det[:, None, None] * self.padding_factor
        max_r2 = _relion_round(self.r_max * self.padding_factor) ** 2
        # Magnification identity (m00 = m11 = 1): Am_* are the first two columns.
        am_xx, am_xy = inv[:, 0, 0] * 1.0 + inv[:, 0, 1] * 0.0, inv[:, 0, 0] * 0.0 + inv[:, 0, 1] * 1.0
        am_yx, am_yy = inv[:, 1, 0] * 1.0 + inv[:, 1, 1] * 0.0, inv[:, 1, 0] * 0.0 + inv[:, 1, 1] * 1.0
        am_zx, am_zy = inv[:, 2, 0] * 1.0 + inv[:, 2, 1] * 0.0, inv[:, 2, 0] * 0.0 + inv[:, 2, 1] * 1.0
        ata_xx = am_xx * am_xx + am_yx * am_yx + am_zx * am_zx
        ata_xy = am_xx * am_xy + am_yx * am_yy + am_zx * am_zy
        ata_yy = am_xy * am_xy + am_yy * am_yy + am_zy * am_zy
        ata_xy2 = ata_xy * ata_xy
        rows = np.arange(s)
        y = np.where(rows < sh, rows, rows - s).astype(np.float64)
        first_allowed = np.where(rows < sh, 0, 1)
        discr = ata_xy2[:, None] * y[None, :] * y[None, :] - ata_xx[:, None] * (
            ata_yy[:, None] * y[None, :] * y[None, :] - max_r2
        )
        with np.errstate(invalid="ignore"):
            d = np.sqrt(discr) / ata_xx[:, None]
        q = -ata_xy[:, None] * y[None, :] / ata_xx[:, None]
        first_x = np.maximum(np.ceil(q - d), first_allowed[None, :])
        last_x = np.minimum(np.floor(q + d), sh - 1)
        x = np.arange(sh, dtype=np.float64)
        take = (
            (discr[:, :, None] >= 0.0)
            & (x[None, None, :] >= first_x[:, :, None])
            & (x[None, None, :] <= last_x[:, :, None])
        )
        take &= weights > 0.0
        image, row, col = np.nonzero(take)
        value = images[image, row, col]
        weight = weights[image, row, col]
        xu = col.astype(np.float64)
        yu = y[row]
        a_inv = inv[image]
        z_on_ewald = 0.0 * (xu * xu + yu * yu)
        xp = a_inv[:, 0, 0] * xu + a_inv[:, 0, 1] * yu + a_inv[:, 0, 2] * z_on_ewald
        yp = a_inv[:, 1, 0] * xu + a_inv[:, 1, 1] * yu + a_inv[:, 1, 2] * z_on_ewald
        zp = a_inv[:, 2, 0] * xu + a_inv[:, 2, 1] * yu + a_inv[:, 2, 2] * z_on_ewald
        keep = (xp * xp + yp * yp + zp * zp) <= max_r2
        negative = xp < 0
        xp, yp, zp = np.where(negative, -xp, xp), np.where(negative, -yp, yp), np.where(negative, -zp, zp)
        value = np.where(negative, np.conj(value), value)
        x0f, y0f, z0f = np.floor(xp), np.floor(yp), np.floor(zp)
        fx, fy, fz = xp - x0f, yp - y0f, zp - z0f
        nz, ny, nx = self.data.shape
        x0 = x0f.astype(np.int64)
        y0 = y0f.astype(np.int64) + ny // 2
        z0 = z0f.astype(np.int64) + nz // 2
        keep &= (x0 >= 0) & (x0 + 1 < nx) & (y0 >= 0) & (y0 + 1 < ny) & (z0 >= 0) & (z0 + 1 < nz)
        x0, y0, z0, fx, fy, fz, value, weight = (v[keep] for v in (x0, y0, z0, fx, fy, fz, value, weight))
        mfx, mfy, mfz = 1.0 - fx, 1.0 - fy, 1.0 - fz
        corners = (
            (0, 0, 0, mfz * mfy * mfx),
            (0, 0, 1, mfz * mfy * fx),
            (0, 1, 0, mfz * fy * mfx),
            (0, 1, 1, mfz * fy * fx),
            (1, 0, 0, fz * mfy * mfx),
            (1, 0, 1, fz * mfy * fx),
            (1, 1, 0, fz * fy * mfx),
            (1, 1, 1, fz * fy * fx),
        )
        size = self.weight.size
        data_real = np.zeros(size)
        data_imag = np.zeros(size)
        weight_sum = np.zeros(size)
        for dz, dy, dx, dd in corners:
            flat = ((z0 + dz) * ny + (y0 + dy)) * nx + (x0 + dx)
            data_real += np.bincount(flat, weights=dd * value.real, minlength=size)
            data_imag += np.bincount(flat, weights=dd * value.imag, minlength=size)
            weight_sum += np.bincount(flat, weights=dd * weight, minlength=size)
        self.data += (data_real + 1j * data_imag).reshape(self.data.shape)
        self.weight += weight_sum.reshape(self.weight.shape)

    def reconstruct(self) -> np.ndarray:
        """``reconstruct(vol, max_iter_preweight, do_map=false, ...)`` with ``skip_gridding`` and ``data_dim = 2``."""

        pf = self.padding_factor
        r_max = self.r_max
        n = self.pad_size
        half = n // 2 + 1
        max_r2 = _relion_round(r_max * pf) * _relion_round(r_max * pf)
        kp = _fftw_axis(n, half)[:, None, None]
        ip = _fftw_axis(n, half)[None, :, None]
        jp = np.arange(half)[None, None, :]
        r2 = kp * kp + ip * ip + jp * jp
        inside = r2 <= max_r2
        centre = n // 2
        source = (kp + centre, ip + centre, np.broadcast_to(jp, r2.shape))
        f_weight = np.where(inside, self.weight[source[0], source[1], source[2]], 0.0)
        f_conv = np.where(inside, self.data[source[0], source[1], source[2]], 0.0)
        # 1/1000 of the radially averaged weight per shell floors the division.
        round_max_r2 = _relion_round(r_max * pf * r_max * pf)
        shell = np.floor(np.sqrt(r2.astype(np.float64)) / pf).astype(np.int64)
        within = r2 < round_max_r2
        radavg = np.bincount(shell[within], weights=f_weight[within], minlength=r_max)[:r_max]
        counter = np.bincount(shell[within], minlength=r_max)[:r_max].astype(np.float64)
        if np.any((counter <= 0) & (radavg <= 0)):
            raise ValueError("BUG: zeros in counter or radavg_weight")
        radavg = radavg / (1000.0 * counter)
        floor = radavg[np.minimum(shell, r_max - 1)]
        weight = np.maximum(f_weight, floor)
        f_conv = np.where(weight == 0.0, f_conv, f_conv / np.where(weight == 0.0, 1.0, weight))
        return self._window_to_oridim_real_space(f_conv)

    def _window_to_oridim_real_space(self, f_in) -> np.ndarray:
        """``windowToOridimRealSpace`` then ``griddingCorrect`` (backprojector.cpp:2848-2967, projector.cpp:595)."""

        pf, ori = self.padding_factor, self.ori_size
        padoridim = _relion_round(pf * ori)
        padoridim += padoridim % 2
        new_half = padoridim // 2 + 1
        n, half = f_in.shape[0], f_in.shape[-1]
        out = np.zeros((padoridim, padoridim, new_half), dtype=np.complex128)
        kp = _fftw_axis(n, half)
        ip = _fftw_axis(n, half)
        jp = np.arange(half)
        if new_half > half:
            max_r2 = (half - 1) * (half - 1)
            r2 = kp[:, None, None] ** 2 + ip[None, :, None] ** 2 + jp[None, None, :] ** 2
            k_idx, i_idx, j_idx = np.nonzero(r2 <= max_r2)
            out[kp[k_idx] % padoridim, ip[i_idx] % padoridim, jp[j_idx]] = f_in[k_idx, i_idx, j_idx]
        else:
            ko = _fftw_axis(padoridim, new_half)
            src = (ko % n)[:, None, None], (ko % n)[None, :, None], np.arange(new_half)[None, None, :]
            out = f_in[src[0], src[1], src[2]]
        sign = (-1.0) ** (np.add.outer(np.add.outer(np.arange(padoridim), np.arange(padoridim)), np.arange(new_half)))
        out = out * sign
        # FFTW's backward transform is unnormalised.
        real = np.fft.irfftn(out, s=(padoridim,) * 3, axes=(0, 1, 2)) * float(padoridim) ** 3
        start = padoridim // 2 + int(-int(float(ori) / 2.0))
        real = real[start : start + ori, start : start + ori, start : start + ori]
        real = real / float(pf * pf * pf * ori)
        real = soft_mask_outside_map(real)
        axes = [_logical_axis(ori)] * 3
        k, i, j = np.meshgrid(*axes, indexing="ij")
        r = np.sqrt((k * k + i * i + j * j).astype(np.float64))
        with np.errstate(invalid="ignore", divide="ignore"):
            rval = r / (ori * pf)
            sinc = np.sin(_PI * rval) / (_PI * rval)
            corrected = real / (sinc * sinc)
        return np.where(r > 0.0, corrected, real)


def bootstrap_references(
    *,
    images,
    ctf_images,
    ori_size: int,
    pixel_size: float,
    nr_classes: int,
    particle_diameter_ang: float,
    width_mask_edge_px: float,
    do_zero_mask: bool,
    random_seed: int,
    padding_factor: int,
    minimum_nr_particles: int,
    particle_seed_ids=None,
    current_size: int = -1,
) -> tuple[np.ndarray, relion_random.GlibcRand | None]:
    """The random-orientation bootstrap reconstruction per class, RELION layout ``[K, N, N, N]``.

    ``images`` are real-space particles in RELION order; ``ctf_images`` is ``None``
    without CTF correction, else each particle's full-size ``[N, N // 2 + 1]`` CTF.
    ``current_size`` (``wsum_model.current_size``; -1 for the full box) windows the
    images, the CTF and the back-projector. Returns the references and the C
    ``rand()`` state the blob draws continue from.
    """

    images = np.asarray(images, dtype=np.float64)
    n_images = images.shape[0]
    todo = max(int(minimum_nr_particles), int(nr_classes) * 5)
    todo = min(todo, n_images)
    seeds = np.arange(todo) if particle_seed_ids is None else np.asarray(particle_seed_ids, dtype=np.int64)
    if seeds.size < todo:
        raise ValueError("particle_seed_ids must contain at least minimum_nr_particles entries")
    radius_px = float(particle_diameter_ang) / (2.0 * float(pixel_size))
    generator = None
    eulers = np.empty((todo, 3))
    for part in range(todo):
        generator = relion_random.GlibcRand(int(random_seed) + int(seeds[part]))
        eulers[part] = [
            float(relion_random.rnd_unif(generator)) * 360.0,
            float(relion_random.rnd_unif(generator)) * 180.0,
            float(relion_random.rnd_unif(generator)) * 360.0,
        ]
    matrices = euler_angles_to_matrix(eulers)
    size = int(ori_size) if current_size <= 0 else int(current_size)
    projectors = [BackProjector3D(ori_size, padding_factor, current_size) for _ in range(int(nr_classes))]
    # windowFourierTransform to the bootstrap size, then CenterFFTbySign on the window
    # (the order matters for odd sizes).
    rows_window = fftw_window_rows(int(ori_size), size)
    sign = (-1.0) ** np.add.outer(np.arange(size), np.arange(size // 2 + 1))
    batch = 64
    for start in range(0, todo, batch):
        rows = np.arange(start, min(start + batch, todo))
        stack = images[rows]
        if do_zero_mask:
            stack = np.stack([soft_mask_outside_map(image, radius_px, float(width_mask_edge_px)) for image in stack])
        fimg = np.fft.rfftn(stack, axes=(-2, -1)) / float(ori_size * ori_size)
        fimg = fimg[:, rows_window, : size // 2 + 1] * sign
        if ctf_images is None:
            weight = np.ones(fimg.shape)
        else:
            ctf = np.asarray(ctf_images[rows], dtype=np.float64)[:, rows_window, : size // 2 + 1]
            fimg = fimg * ctf
            weight = ctf * ctf
        for k in range(int(nr_classes)):
            mine = (rows % int(nr_classes)) == k
            if np.any(mine):
                projectors[k].backproject(fimg[mine], matrices[rows[mine]], weight[mine])
    return np.asarray([projector.reconstruct() for projector in projectors]), generator


def _random_normal(generator: relion_random.GlibcRand) -> float:
    """``SomGraph::random_normal<double>`` (gradient_optimisation.h:390-395)."""

    v1 = (float(generator.rand()) + 1.0) / (float(relion_random.RAND_MAX) + 1.0)
    v2 = (float(generator.rand()) + 1.0) / (float(relion_random.RAND_MAX) + 1.0)
    return math.cos(2 * 3.14 * v2) * math.sqrt(-2.0 * math.log(v1))


def _make_blobs_3d(amplitude, nr_blobs: int, diameter: float, generator) -> np.ndarray:
    """``SomGraph::make_blobs_3d`` (not helical) drawing from RELION's ``rand()`` stream."""

    amp = np.asarray(amplitude, dtype=np.float64)
    zdim, ydim, xdim = amp.shape
    box = np.zeros_like(amp)
    blobs = []
    for _ in range(int(nr_blobs)):
        bx = _random_normal(generator) * diameter / 6.0 + xdim / 2.0
        by = _random_normal(generator) * diameter / 6.0 + ydim / 2.0
        bz = _random_normal(generator) * diameter / 6.0 + zdim / 2.0
        if bz < zdim and by < ydim and bx < xdim:
            value = abs(amp.reshape(-1)[int(bz) * ydim * xdim + int(by) * xdim + int(bx)])
        else:
            value = 0.0
        blobs.append((bx, by, bz, value))
    span = diameter / 3.0
    sigma_inv = 10.0 / diameter
    for bx, by, bz, value in blobs:
        if not value > 0:
            continue
        # for (int z = MAX(0, bz - span); z < MIN(zdim, bz + span); z++): int(...) truncates.
        z = np.arange(int(max(0, bz - span)), zdim)
        z = z[z < min(zdim, bz + span)]
        y = np.arange(int(max(0, by - span)), ydim)
        y = y[y < min(ydim, by + span)]
        x = np.arange(int(max(0, bx - span)), xdim)
        x = x[x < min(xdim, bx + span)]
        zp = (z - bz) * sigma_inv
        yp = (y - by) * sigma_inv
        xp = (x - bx) * sigma_inv
        exponent = (
            -xp[None, None, :] * xp[None, None, :]
            - yp[None, :, None] * yp[None, :, None]
            - zp[:, None, None] * zp[:, None, None]
        )
        box[np.ix_(z, y, x)] += np.exp(exponent) * value
    return box


def _std(values) -> float:
    """``SomGraph::std``: population standard deviation, summed serially."""

    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    mean = np.cumsum(flat)[-1] / flat.size
    diff = flat - mean
    return math.sqrt(np.cumsum(diff * diff)[-1] / float(flat.size))


def postprocess_references(
    references,
    *,
    generator,
    pixel_size: float,
    ini_high_ang: float,
    particle_diameter_ang: float,
    width_mask_edge_px: float,
    do_init_blobs: bool = True,
) -> np.ndarray:
    """Low-pass, blobs and soft mask of the bootstrap references (ml_optimiser.cpp:2940-2980)."""

    refs = np.asarray(references, dtype=np.float64)
    ori_size = refs.shape[-1]
    diameter_px = float(particle_diameter_ang) / float(pixel_size)
    out = []
    for vol in refs:
        vol = initial_low_pass_filter(vol, ori_size, pixel_size, ini_high_ang)
        if do_init_blobs:
            blobs_pos = _make_blobs_3d(vol, 40, diameter_px, generator)
            blobs_neg = _make_blobs_3d(vol, 40, diameter_px, generator)
            old_std = _std(vol)
            vol = blobs_pos - blobs_neg / 2.0
            new_std = _std(vol)
            if new_std > 0.0:
                vol = vol * (old_std / new_std)
            vol = initial_low_pass_filter(vol, ori_size, pixel_size, ini_high_ang)
            vol = soft_mask_outside_map(vol, diameter_px / 2.0, float(width_mask_edge_px))
        out.append(vol)
    return np.asarray(out)


__all__ = [
    "BackProjector3D",
    "bootstrap_references",
    "initial_low_pass_filter",
    "postprocess_references",
    "soft_mask_outside_map",
]
