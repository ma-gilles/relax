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
The projections, shifts and per-step SNR sums run in float64 on the default JAX
device, every trial image of a chunk at once; the projection matrices, the step
schedule and the stopping rule stay on the host. The SNR of a trial is summed over
its images and pixels in the device's reduction order rather than RELION's serial
order (float64 rounding, far below the ``pvalue`` step).
The device arrays take the stable Fourier-window class of the current size
(:func:`_capacity_size`): the images are laid out in the class's FFTW half grid and
the projector slab is zero-padded to the class radius, so the device programs
compile once per class instead of once per current size (13 sizes in a 200-iteration
VDAM run). The pixels outside the current size hold zeros and are not valid SNR terms.
A slab larger than the device budget
(:func:`relax.sparse_pass2.sparse_pass2_budget.accuracy_slab_resident_bytes`) is
streamed in z-plane chunks of at most ``ACCURACY_SLAB_STREAM_CHUNK_BYTES``; each
sample takes each of its two z planes from the chunk holding it, so the values do
not depend on the chunking.
``relax.relion_bind`` is the unit-test oracle
(``tests/unit/test_relion_expected_accuracy_vs_relion_bind.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from relax.helpers import relion_random
from relax.helpers.fourier_window import stable_fourier_window_current_size, stable_fourier_window_quantum
from relax.sparse_pass2.sparse_pass2_budget import ACCURACY_SLAB_STREAM_CHUNK_BYTES, accuracy_slab_resident_bytes

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


def _capacity_size(size: int, full_size: int) -> int:
    """The stable Fourier-window class of ``size`` in a box of ``full_size`` (a shape, never a cutoff)."""

    size, full_size = int(size), int(full_size)
    if size <= 0 or size >= full_size or size % 2 or full_size % 2:
        return size
    return stable_fourier_window_current_size(size, full_size, quantum=stable_fourier_window_quantum())


def _capacity_rows(size: int, capacity: int) -> np.ndarray:
    """The row of each FFTW row of a ``size`` image in the ``capacity`` image (same frequency)."""

    rows = np.arange(size)
    return np.where(rows <= size // 2, rows, rows - size + capacity)


def _to_capacity(values, capacity: int, fill) -> np.ndarray:
    """``[..., size, size // 2 + 1]`` FFTW half images in the ``capacity`` layout, ``fill`` elsewhere."""

    size, half = values.shape[-2:]
    if capacity == size:
        return values
    out = np.full(values.shape[:-2] + (capacity, capacity // 2 + 1), fill, dtype=values.dtype)
    out[..., _capacity_rows(size, capacity), :half] = values
    return out


def _visited_pixels(size: int):
    """Rows, columns and signed row frequencies of the pixels ``Projector::project`` visits."""

    half = size // 2 + 1
    r_max_out = half - 1
    rows = np.arange(size)
    y = np.where(rows <= r_max_out, rows, rows - size)
    x_max = np.floor(np.sqrt((r_max_out * r_max_out - y * y).astype(np.float64))).astype(np.int64)
    # Row i, x in [0, FLOOR(sqrt(r_max_out^2 - y^2))].
    row_index, x_index = np.nonzero(np.arange(half)[None, :] <= x_max[:, None])
    return row_index, x_index, y[row_index]


class _Projector:
    """A ``Projector`` of 3-D ``data`` (``[L, L, L // 2 + 1]``, Xmipp origin) at ``r_max``.

    ``capacity_image_size`` and ``capacity_r_max`` give the device shapes: the images take the
    capacity's FFTW half layout and the slab is zero-padded to the capacity radius. The
    visited pixels and the radius test stay those of ``image_size`` and ``r_max``.
    The float64 real and imaginary planes stay on the host; :meth:`device` places them.
    """

    def __init__(
        self, data, r_max: int, padding_factor: int, image_size: int,
        *, capacity_image_size: int | None = None, capacity_r_max: int | None = None,
    ):
        data = np.asarray(data)
        self.padding_factor = float(padding_factor)
        self.r_max_ref_2 = float(int(r_max * self.padding_factor) ** 2)
        if capacity_r_max is not None:
            # Projector::data of a larger r_max is this slab inside more zeros.
            grow = 2 * (int(capacity_r_max * self.padding_factor) + 1) + 1 - data.shape[0]
            if grow > 0 and grow % 2 == 0:
                data = np.pad(data, ((grow // 2,) * 2, (grow // 2,) * 2, (0, grow // 2)))
        self.data_shape = tuple(int(v) for v in data.shape)
        self.real = np.ascontiguousarray(data.real, dtype=np.float64)
        self.imag = np.ascontiguousarray(data.imag, dtype=np.float64)
        del data
        size = int(image_size)
        capacity = size if capacity_image_size is None else int(capacity_image_size)
        half = capacity // 2 + 1
        row_index, x_index, y = _visited_pixels(size)
        # Pixels past the logical count project the origin into a spare column that is dropped.
        spare = _visited_pixels(capacity)[0].size - row_index.size
        self.pixel_flat = np.concatenate(
            [_capacity_rows(size, capacity)[row_index] * half + x_index, np.full(spare, capacity * half)]
        )
        self.pixel_x = np.concatenate([x_index.astype(np.float64), np.zeros(spare)])
        self.pixel_y = np.concatenate([y.astype(np.float64), np.zeros(spare)])
        self.shape = (capacity, half)

    def a_inv(self, matrices):
        """``A^-1 * padding_factor`` of ``[M, 3, 3]`` ``A`` (host float64, matrix2d.h:1108-1123)."""

        return _inverse3(np.asarray(matrices, dtype=np.float64)) * self.padding_factor

    def device(self, resident_bytes: int | None, stream_chunk_bytes: int):
        """The slab placed for :func:`_project_on_device`.

        A slab within ``resident_bytes`` (or any slab when it is None) is one resident
        chunk. A larger one is cut into equal z-plane chunks of at most
        ``stream_chunk_bytes`` (the last zero-padded, so one program serves them all)
        that each projection uploads in turn.
        """

        nz, ny, nx = self.data_shape
        plane_bytes = 2 * 8 * ny * nx
        resident = resident_bytes is None or nz * plane_bytes <= int(resident_bytes)
        planes = nz if resident else max(1, min(nz, int(stream_chunk_bytes) // plane_bytes))
        pixels = tuple(jnp.asarray(v) for v in (self.pixel_x, self.pixel_y, self.pixel_flat))
        if resident:
            whole = (jnp.asarray(self.real.reshape(-1)), jnp.asarray(self.imag.reshape(-1)))
            return _DeviceSlab(pixels, planes, ((0, whole),))
        chunks = []
        for z_lo in range(0, nz, planes):
            real, imag = self.real[z_lo : z_lo + planes], self.imag[z_lo : z_lo + planes]
            if real.shape[0] < planes:
                pad = ((0, planes - real.shape[0]), (0, 0), (0, 0))
                real, imag = np.pad(real, pad), np.pad(imag, pad)
            chunks.append((z_lo, (real.reshape(-1), imag.reshape(-1))))
        return _DeviceSlab(pixels, planes, tuple(chunks))


@dataclass(frozen=True)
class _DeviceSlab:
    """Visited pixels on the device and the slab's z-plane chunks (``(z_lo, (real, imag))``).

    One chunk holds the whole slab and is already on the device; several chunks are
    host arrays uploaded by each projection.
    """

    pixels: tuple
    planes: int
    chunks: tuple

    @property
    def streamed(self) -> bool:
        return len(self.chunks) > 1


def _project_on_device(slab: _DeviceSlab, a_inv, *, data_shape, n_full, r_max_ref_2):
    """``get2DFourierTransform(F, A)`` for ``[M, 3, 3]`` ``a_inv``; ``[M, H * W]`` real and imag.

    ``Projector::project`` (projector.cpp:630-790) at the pixels it visits: the
    rotated coordinate, the radius test, the Hermitian flip for ``xp < 0`` and
    RELION's nested ``LIN_INTERP`` (projector.cpp:733-740). The two z planes of
    every sample are interpolated in x and y by the chunk that holds each plane,
    then in z, so a streamed slab gives the resident slab's values bit for bit.
    """

    pixel_x, pixel_y, pixel_flat = slab.pixels
    a_inv = jnp.asarray(a_inv)
    sample = _sample_coordinates(pixel_x, pixel_y, a_inv, data_shape=data_shape, r_max_ref_2=r_max_ref_2)
    planes = jnp.zeros((4,) + sample[0].shape, dtype=jnp.float64)
    for z_lo, (real, imag) in slab.chunks:
        planes = _interpolate_planes(
            jnp.asarray(real), jnp.asarray(imag), np.int64(z_lo), planes, *sample,
            plane_shape=(slab.planes,) + tuple(data_shape[1:]),
        )
    return _finish_projection(planes, sample[0], sample[1], sample[4], pixel_flat, n_full=n_full)


@partial(jax.jit, static_argnames=("data_shape",))
def _sample_coordinates(pixel_x, pixel_y, a_inv, *, data_shape, r_max_ref_2):
    """Each sample's radius/bounds test, Hermitian flip, z plane, in-plane offset and fractions."""

    x, y = pixel_x[None, :], pixel_y[None, :]
    xp = a_inv[:, 0, 0, None] * x + a_inv[:, 0, 1, None] * y
    yp = a_inv[:, 1, 0, None] * x + a_inv[:, 1, 1, None] * y
    zp = a_inv[:, 2, 0, None] * x + a_inv[:, 2, 1, None] * y
    inside = (xp * xp + yp * yp + zp * zp) <= r_max_ref_2
    negative = xp < 0
    xp = jnp.where(negative, -xp, xp)
    yp = jnp.where(negative, -yp, yp)
    zp = jnp.where(negative, -zp, zp)
    x0 = jnp.floor(xp)
    y0 = jnp.floor(yp)
    z0 = jnp.floor(zp)
    fx, fy, fz = xp - x0, yp - y0, zp - z0
    nz, ny, nx = data_shape
    x0 = x0.astype(jnp.int64)
    y0 = y0.astype(jnp.int64) + ny // 2
    z0 = z0.astype(jnp.int64) + nz // 2
    inside &= (x0 >= 0) & (x0 + 1 < nx) & (y0 >= 0) & (y0 + 1 < ny) & (z0 >= 0) & (z0 + 1 < nz)
    in_plane = jnp.where(inside, y0 * nx + x0, 0)
    return inside, negative, z0, in_plane, fz, fx, fy


@partial(jax.jit, static_argnames=("plane_shape",), donate_argnums=(3,))
def _interpolate_planes(real, imag, z_lo, planes, inside, negative, z0, in_plane, fz, fx, fy, *, plane_shape):
    """``planes`` (real z0, real z0+1, imag z0, imag z0+1) with the samples whose plane is in this chunk."""

    n_planes, ny, nx = plane_shape
    step_y, step_z = nx, ny * nx

    def bilinear(values, base):
        d00, d01 = values[base], values[base + 1]
        d10, d11 = values[base + step_y], values[base + step_y + 1]
        dx0 = d00 + (d01 - d00) * fx
        dx1 = d10 + (d11 - d10) * fx
        return dx0 + (dx1 - dx0) * fy

    out = []
    for values in (real, imag):
        for dz in (0, 1):
            local = z0 + dz - z_lo
            take = inside & (local >= 0) & (local < n_planes)
            base = jnp.where(take, local * step_z + in_plane, 0)
            out.append((take, bilinear(values, base)))
    return jnp.stack([jnp.where(take, value, planes[i]) for i, (take, value) in enumerate(out)])


@partial(jax.jit, static_argnames=("n_full",))
def _finish_projection(planes, inside, negative, fz, pixel_flat, *, n_full):
    """The z interpolation of the two planes, the flip of the imaginary part and the pixel layout."""

    real0, real1, imag0, imag1 = planes
    out_real = jnp.where(inside, real0 + (real1 - real0) * fz, 0.0)
    out_imag = imag0 + (imag1 - imag0) * fz
    out_imag = jnp.where(inside, jnp.where(negative, -out_imag, out_imag), 0.0)
    zeros = jnp.zeros((inside.shape[0], n_full + 1), dtype=jnp.float64)
    return zeros.at[:, pixel_flat].set(out_real)[:, :n_full], zeros.at[:, pixel_flat].set(out_imag)[:, :n_full]


@partial(jax.jit, static_argnames=("shape", "oridim"))
def _shift(real, imag, xshift, yshift, *, shape, oridim):
    """``shiftImageInFourierTransform(in, out, oridim, xshift, yshift)`` in 2-D for ``[M, H * W]`` images (fftw.cpp:842-906)."""

    size, half = shape
    xs = xshift / -float(oridim)
    ys = yshift / -float(oridim)
    unchanged = (jnp.abs(xs) < _EQUAL_ACCURACY) & (jnp.abs(ys) < _EQUAL_ACCURACY)
    rows = jnp.arange(size)
    y = jnp.where(rows < half, rows, rows - size).astype(jnp.float64)
    x = jnp.arange(half, dtype=jnp.float64)
    dotp = 2 * _PI * (x[None, None, :] * xs[:, None, None] + y[None, :, None] * ys[:, None, None])
    b, a = jnp.sin(dotp).reshape(-1, size * half), jnp.cos(dotp).reshape(-1, size * half)
    ac = a * real
    bd = b * imag
    ab_cd = (a + b) * (real + imag)
    out_real = jnp.where(unchanged[:, None], real, ac - bd)
    out_imag = jnp.where(unchanged[:, None], imag, ab_cd - ac - bd)
    return out_real, out_imag


@partial(jax.jit, static_argnames=("n_trials",))
def _trial_snr(f1_real, f1_imag, f2_real, f2_imag, ctf, valid, denominator, image_trial, *, n_trials):
    """``my_snr += norm(F1 - F2) / (2 fudge sigma)`` of each trial over its images' pixels."""

    if ctf is not None:
        f2_real, f2_imag = f2_real * ctf, f2_imag * ctf
    diff_real, diff_imag = f1_real - f2_real, f1_imag - f2_imag
    terms = jnp.where(valid[None, :], (diff_real * diff_real + diff_imag * diff_imag) / denominator[None, :], 0.0)
    return jax.ops.segment_sum(jnp.sum(terms, axis=1), image_trial, num_segments=n_trials)


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


_CHUNK_VALUES = 1 << 25


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


def _chunk_errors(projector, device, matrices, *, rows, counts, eulers, draws, ctf, aproj, image_full_size, valid, denominator):
    """Angular and translational errors of one chunk of trials (``rows`` are their images).

    Every image of the chunk is projected and scored at each step; a trial's error is
    the first step whose SNR is not ``<= pvalue`` (``while (my_snr <= pvalue)`` stops on
    anything else, NaN included), as when only the still-searching trials are scored.
    """

    n_trials = counts.size
    image_trial = jnp.asarray(np.repeat(np.arange(n_trials), counts))
    project = partial(
        _project_on_device, device,
        data_shape=projector.data_shape,
        n_full=int(projector.shape[0] * projector.shape[1]),
        r_max_ref_2=np.float64(projector.r_max_ref_2),
    )
    ctf = None if ctf is None else jnp.asarray(ctf.reshape(ctf.shape[0], -1))
    valid = jnp.asarray(valid.reshape(-1))
    denominator = jnp.asarray(denominator.reshape(-1))
    snr_of = partial(_trial_snr, ctf=ctf, valid=valid, denominator=denominator, image_trial=image_trial, n_trials=n_trials)
    f1_real, f1_imag = project(jnp.asarray(projector.a_inv(matrices(eulers, rows))))
    f1_ctf_real, f1_ctf_imag = (f1_real, f1_imag) if ctf is None else (f1_real * ctf, f1_imag * ctf)
    column = np.where(draws < 0.3333, 0, np.where(draws < 0.6667, 1, 2))
    errors = []
    for mode in (0, 1):
        angles, shifts, n_scored = _schedule(mode)
        result = np.full(n_trials, angles[n_scored] if mode == 0 else shifts[n_scored])
        active = np.ones(n_trials, dtype=bool)
        for step in range(n_scored):
            if not active.any():
                break
            if mode == 0:
                perturbed = eulers.copy()
                perturbed[np.arange(eulers.shape[0]), column] += angles[step]
                f2_real, f2_imag = project(jnp.asarray(projector.a_inv(matrices(perturbed, rows))))
            else:
                if aproj is not None:
                    # Experiment::getTranslationInTiltSeries (exp_model.cpp:105-113).
                    x3 = np.where(draws < 0.3333, shifts[step], 0.0)
                    y3 = np.where((draws >= 0.3333) & (draws < 0.6667), shifts[step], 0.0)
                    z3 = np.where(draws >= 0.6667, shifts[step], 0.0)
                    xshift = aproj[:, 0, 0] * x3 + aproj[:, 0, 1] * y3 + aproj[:, 0, 2] * z3
                    yshift = aproj[:, 1, 0] * x3 + aproj[:, 1, 1] * y3 + aproj[:, 1, 2] * z3
                else:
                    xshift = np.where(draws < 0.5, shifts[step], 0.0)
                    yshift = np.where(draws < 0.5, 0.0, shifts[step])
                f2_real, f2_imag = _shift(
                    f1_real, f1_imag, jnp.asarray(-xshift), jnp.asarray(-yshift),
                    shape=projector.shape, oridim=int(image_full_size),
                )
            snr = np.asarray(snr_of(f1_ctf_real, f1_ctf_imag, f2_real, f2_imag))
            done = active & ~(snr <= PVALUE)
            result[done] = angles[step] if mode == 0 else shifts[step]
            active &= ~done
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
    projection_left=None,
) -> ExpectedErrors:
    """``calculateExpectedAngularErrors`` over the trials, as the binding's oracle computes it.

    ``projector_data`` ``[K, L, L, L // 2 + 1]`` is each class's ``Projector::data``
    at ``projector_r_max``. ``ctf_images`` is ``None`` without CTF correction, else
    one ``[current_image_size, current_image_size // 2 + 1]`` CTF per image of the
    trial particles (in trial-image order). Tilt images: ``image_offsets``
    ``[n_particles + 1]`` and ``image_projections`` ``[n_images, 3, 3]`` (``Aproj``).
    ``projection_left`` (3x3, ``ObservationModel::applyAnisoMag``'s ``inv(M3)``)
    multiplies both projection matrices on the left (ml_optimiser.cpp:9526, 9589).
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
    # Device shapes: the stable window class of the image and of the projector radius.
    capacity = _capacity_size(int(current_image_size), image_full_size)
    capacity_r_max = _capacity_size(2 * int(projector_r_max), int(ori_size)) // 2
    valid, denominator = _to_capacity(valid, capacity, False), _to_capacity(denominator, capacity, 1.0)
    ctf = None if ctf_images is None else _to_capacity(np.asarray(ctf_images, dtype=np.float64), capacity, 0.0)
    left = None if projection_left is None else np.asarray(projection_left, dtype=np.float64).reshape(3, 3)
    image_eulers = eulers[image_trial]

    def matrices(rows_eulers, rows):
        a = _euler_matrices(rows_eulers)
        if tomo:
            a = _matmul3(aproj[rows], a)
        if left is not None:
            a = _matmul3(left, a)
        return a * scale_difference

    acc_rot_class = np.full(n_classes, 999.0)
    acc_trans_class = np.full(n_classes, 999.0)
    class_counts = np.zeros(n_classes, dtype=np.int64)
    acc_rot, acc_trans = 999.0, 999.0
    # Trials are independent; a chunk bounds the per-step image arrays.
    n_pixels = capacity * (capacity // 2 + 1)
    chunks = _trial_chunks(counts, max(1, _CHUNK_VALUES // n_pixels))
    for k in range(n_classes):
        if pdf[k] < 0.01:
            continue
        projector = _Projector(
            projector_data[k], int(projector_r_max), int(padding_factor), int(current_image_size),
            capacity_image_size=capacity, capacity_r_max=capacity_r_max,
        )
        device = projector.device(accuracy_slab_resident_bytes(), ACCURACY_SLAB_STREAM_CHUNK_BYTES)
        errors = [np.empty(n_trials), np.empty(n_trials)]
        for chunk in chunks:
            rows = np.concatenate([np.arange(trial_image_offsets[t], trial_image_offsets[t + 1]) for t in chunk])
            chunk_errors = _chunk_errors(
                projector,
                device,
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
